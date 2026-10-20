
from pathlib import Path

import numpy as np
import pydicom

from .dicom_series import DicomSeries
from .dicom_slice import DicomSlice


class DicomSeriesReader:
    """Read and validate a DICOM series from a directory."""

    def read(self, series_path: str | Path) -> DicomSeries:
        series_path = Path(series_path)

        if not series_path.is_dir():
            raise FileNotFoundError(
                f"DICOM series directory not found: {series_path}"
            )

        slices: list[DicomSlice] = []
        skipped_files: list[str] = []
        reference_study_uid: str | None = None
        reference_series_uid: str | None = None
        reference_shape: tuple[int, int] | None = None

        for file_path in sorted(series_path.iterdir()):
            if not file_path.is_file():
                continue

            try:
                dataset = pydicom.dcmread(
                    file_path,
                    force=True,
                )

                if not hasattr(dataset, "PixelData"):
                    continue

                pixel_array = dataset.pixel_array

                if pixel_array.ndim != 2:
                    raise ValueError(
                        f"Expected a 2D image; got shape "
                        f"{pixel_array.shape}"
                    )

                if not np.issubdtype(
                    pixel_array.dtype, np.number
                ):
                    raise ValueError("Pixel array is not numeric.")

                if not np.all(np.isfinite(pixel_array)):
                    raise ValueError("Pixel array contains non-finite values.")

                study_uid = str(
                    getattr(dataset, "StudyInstanceUID", "")
                )
                series_uid = str(
                    getattr(dataset, "SeriesInstanceUID", "")
                )

                if not study_uid or not series_uid:
                    raise ValueError(
                        "Missing StudyInstanceUID or SeriesInstanceUID."
                    )

                # The directory must contain one study and one series.
                if reference_study_uid is None:
                    reference_study_uid = study_uid
                    reference_series_uid = series_uid
                    reference_shape = pixel_array.shape
                elif (
                    study_uid != reference_study_uid
                    or series_uid != reference_series_uid
                ):
                    raise ValueError(
                        "DICOM file belongs to a different study or series."
                    )

                if pixel_array.shape != reference_shape:
                    raise ValueError(
                        f"Inconsistent image shape {pixel_array.shape}; "
                        f"expected {reference_shape}."
                    )

                instance_value = getattr(
                    dataset, "InstanceNumber", None
                )
                if instance_value is None:
                    raise ValueError("Missing InstanceNumber.")

                instance_number = int(instance_value)

                slices.append(
                    DicomSlice(
                        path=file_path,
                        instance_number=instance_number,
                        image_position=self._get_image_position(dataset),
                        image_orientation=self._get_image_orientation(dataset),
                        pixel_array=np.ascontiguousarray(pixel_array),
                    )
                )

            except Exception as exc:
                skipped_files.append(
                    f"{file_path.name}: {type(exc).__name__}: {exc}"
                )

        if skipped_files:
            print(
                f"Skipped {len(skipped_files)} file(s) in "
                f"{series_path}:"
            )
            for message in skipped_files:
                print(f"  - {message}")

        if not slices:
            raise ValueError(
                f"No valid DICOM images found in {series_path}"
            )

        # A mixed series is an error, not something to silently ignore.
        # If any file was rejected for a study/series mismatch, stop.
        if any(
            "different study or series" in message
            for message in skipped_files
        ):
            raise ValueError(
                f"Mixed study/series content detected in {series_path}. "
                "Inspect the skipped-file messages."
            )

        return DicomSeries(
            study_instance_uid=reference_study_uid,
            series_instance_uid=reference_series_uid,
            slices=tuple(slices),
        )

    @staticmethod
    def _get_image_position(
        dataset,
    ) -> tuple[float, float, float] | None:
        position = getattr(dataset, "ImagePositionPatient", None)

        if position is None or len(position) != 3:
            return None

        result = tuple(float(value) for value in position)

        if not np.all(np.isfinite(result)):
            return None

        return result

    @staticmethod
    def _get_image_orientation(
        dataset,
    ) -> tuple[
        float, float, float,
        float, float, float
    ] | None:
        orientation = getattr(
            dataset, "ImageOrientationPatient", None
        )

        if orientation is None or len(orientation) != 6:
            return None

        result = tuple(float(value) for value in orientation)

        if not np.all(np.isfinite(result)):
            return None

        return result
