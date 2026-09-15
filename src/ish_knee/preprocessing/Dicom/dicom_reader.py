from pathlib import Path

import pydicom

from .dicom_series import DicomSeries
from .dicomslice import DicomSlice


class DicomSeriesReader:
    """
    Reads a DICOM series from a directory.
    """

    def read(
        self,
        series_path: str | Path,
    ) -> DicomSeries:

        series_path = Path(series_path)

        if not series_path.is_dir():
            raise FileNotFoundError(
                f"DICOM series directory not found: "
                f"{series_path}"
            )

        slices: list[DicomSlice] = []

        # --------------------------------------------------------------
        # Read every file in the series directory
        # --------------------------------------------------------------

        for file_path in sorted(series_path.iterdir()):

            if not file_path.is_file():
                continue

            try:
                dataset = pydicom.dcmread(
                    file_path,
                    force=True,
                )

                # Make sure this is an image that contains
                # pixel data.
                if not hasattr(dataset, "PixelData"):
                    continue

                pixel_array = dataset.pixel_array

            except Exception:
                # Ignore files that cannot be read as DICOM images.
                continue

            # ----------------------------------------------------------
            # Instance number
            # ----------------------------------------------------------

            instance_number = int(
                getattr(
                    dataset,
                    "InstanceNumber",
                    0,
                )
            )

            # ----------------------------------------------------------
            # Image position
            # ----------------------------------------------------------

            image_position = (
                self._get_image_position(dataset)
            )

            # ----------------------------------------------------------
            # Create DicomSlice
            # ----------------------------------------------------------

            slices.append(
                DicomSlice(
                    path=file_path,
                    instance_number=instance_number,
                    image_position=image_position,
                    pixel_array=pixel_array,
                )
            )

        # --------------------------------------------------------------
        # Make sure at least one image was read
        # --------------------------------------------------------------

        if not slices:
            raise ValueError(
                f"No readable DICOM images found in "
                f"{series_path}"
            )

        # --------------------------------------------------------------
        # Get StudyInstanceUID and SeriesInstanceUID
        # --------------------------------------------------------------

        first_dataset = pydicom.dcmread(
            slices[0].path,
            stop_before_pixels=True,
            force=True,
        )

        study_instance_uid = str(
            getattr(
                first_dataset,
                "StudyInstanceUID",
                "",
            )
        )

        series_instance_uid = str(
            getattr(
                first_dataset,
                "SeriesInstanceUID",
                "",
            )
        )

        # --------------------------------------------------------------
        # Create DicomSeries
        # --------------------------------------------------------------

        return DicomSeries(
            study_instance_uid=study_instance_uid,
            series_instance_uid=series_instance_uid,
            slices=tuple(slices),
        )

    @staticmethod
    def _get_image_position(
        dataset,
    ) -> tuple[float, float, float] | None:

        position = getattr(
            dataset,
            "ImagePositionPatient",
            None,
        )

        if position is None:
            return None

        if len(position) < 3:
            return None

        return (
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )