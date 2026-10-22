
import logging
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from .dicom_series import DicomSeries
from .dicom_slice import DicomSlice


logger = logging.getLogger(__name__)


class DicomSeriesReader:
    """
    Reads a directory of DICOM images into a DicomSeries.

    Captures pixel data and geometry metadata required for:
      - DICOM slice ordering
      - 3D volume construction
      - Anatomical orientation analysis
      - Voxel-to-patient affine calculation
      - Geometry-aware resampling
    """

    def read(self, series_path: Path | str) -> DicomSeries:
        """
        Read all valid DICOM slices in a series directory.

        Parameters
        ----------
        series_path:
            Directory containing DICOM image files.

        Returns
        -------
        DicomSeries
            A series containing validated DICOM slices.

        Raises
        ------
        FileNotFoundError
            If the directory does not exist.
        ValueError
            If no valid slices remain or metadata is inconsistent.
        """

        series_path = Path(series_path)

        if not series_path.exists():
            raise FileNotFoundError(
                f"DICOM directory does not exist: {series_path}"
            )

        if not series_path.is_dir():
            raise NotADirectoryError(
                f"Expected a directory: {series_path}"
            )

        files = sorted(
            path for path in series_path.rglob("*")
            if path.is_file()
        )

        if not files:
            raise ValueError(
                f"No files found in DICOM directory: {series_path}"
            )

        slices: list[DicomSlice] = []

        expected_study_uid: str | None = None
        expected_series_uid: str | None = None
        expected_shape: tuple[int, ...] | None = None
        expected_spacing: tuple[float, float] | None = None

        for file_path in files:
            try:
                ds = pydicom.dcmread(
                    str(file_path),
                    force=False,
                )

                # A valid image slice must contain pixel data.
                if not hasattr(ds, "PixelData"):
                    logger.debug(
                        "Skipping file without PixelData: %s",
                        file_path,
                    )
                    continue

                if not hasattr(ds, "StudyInstanceUID"):
                    raise ValueError("Missing StudyInstanceUID")

                if not hasattr(ds, "SeriesInstanceUID"):
                    raise ValueError("Missing SeriesInstanceUID")

                if not hasattr(ds, "ImagePositionPatient"):
                    raise ValueError("Missing ImagePositionPatient")

                if not hasattr(ds, "ImageOrientationPatient"):
                    raise ValueError("Missing ImageOrientationPatient")

                if not hasattr(ds, "PixelSpacing"):
                    raise ValueError("Missing PixelSpacing")

                if not hasattr(ds, "InstanceNumber"):
                    raise ValueError("Missing InstanceNumber")

                study_uid = str(ds.StudyInstanceUID)
                series_uid = str(ds.SeriesInstanceUID)

                image_position = tuple(
                    float(value)
                    for value in ds.ImagePositionPatient
                )

                image_orientation = tuple(
                    float(value)
                    for value in ds.ImageOrientationPatient
                )

                pixel_spacing = tuple(
                    float(value)
                    for value in ds.PixelSpacing
                )

                if len(image_position) != 3:
                    raise ValueError(
                        "ImagePositionPatient must have 3 values"
                    )

                if len(image_orientation) != 6:
                    raise ValueError(
                        "ImageOrientationPatient must have 6 values"
                    )

                if len(pixel_spacing) != 2:
                    raise ValueError(
                        "PixelSpacing must have 2 values"
                    )

                if not np.all(np.isfinite(image_position)):
                    raise ValueError("Invalid image position values")

                if not np.all(np.isfinite(image_orientation)):
                    raise ValueError("Invalid image orientation values")

                if not np.all(np.isfinite(pixel_spacing)):
                    raise ValueError("Invalid pixel spacing values")

                if any(value <= 0 for value in pixel_spacing):
                    raise ValueError("Pixel spacing must be positive")

                instance_number = int(ds.InstanceNumber)

                # Decode the image pixels.
                pixel_array = np.asarray(ds.pixel_array)

                if pixel_array.ndim != 2:
                    raise ValueError(
                        "Expected a 2D image, "
                        f"received shape {pixel_array.shape}"
                    )

                if pixel_array.size == 0:
                    raise ValueError("Pixel array is empty")

                if not np.all(np.isfinite(pixel_array)):
                    raise ValueError(
                        "Pixel array contains non-finite values"
                    )

                # Establish the expected series metadata from
                # the first valid image.
                if expected_study_uid is None:
                    expected_study_uid = study_uid
                    expected_series_uid = series_uid
                    expected_shape = pixel_array.shape
                    expected_spacing = pixel_spacing

                # Reject files that do not belong to this series.
                if study_uid != expected_study_uid:
                    raise ValueError(
                        "StudyInstanceUID differs from the first slice"
                    )

                if series_uid != expected_series_uid:
                    raise ValueError(
                        "SeriesInstanceUID differs from the first slice"
                    )

                if pixel_array.shape != expected_shape:
                    raise ValueError(
                        f"Inconsistent image shape: {pixel_array.shape}; "
                        f"expected {expected_shape}"
                    )

                if not np.allclose(
                    pixel_spacing,
                    expected_spacing,
                    rtol=1e-4,
                    atol=1e-5,
                ):
                    raise ValueError(
                        f"Inconsistent PixelSpacing: {pixel_spacing}; "
                        f"expected {expected_spacing}"
                    )

                dicom_slice = DicomSlice(
                    path=file_path,
                    instance_number=instance_number,
                    image_position=image_position,
                    image_orientation=image_orientation,
                    pixel_spacing=pixel_spacing,
                    pixel_array=pixel_array,
                )

                slices.append(dicom_slice)

            except (InvalidDicomError, OSError, EOFError) as exc:
                logger.warning(
                    "Skipping unreadable DICOM file %s: %s",
                    file_path,
                    exc,
                )

            except Exception as exc:
                logger.warning(
                    "Skipping invalid file %s: %s",
                    file_path,
                    exc,
                )

        if not slices:
            raise ValueError(
                f"No valid DICOM slices found in: {series_path}"
            )

        # Confirm the series metadata was initialized.
        if expected_study_uid is None or expected_series_uid is None:
            raise ValueError(
                f"Could not determine series metadata: {series_path}"
            )

        logger.info(
            "Read %d DICOM slices from %s",
            len(slices),
            series_path,
        )

        logger.info(
            "Study UID: %s | Series UID: %s",
            expected_study_uid,
            expected_series_uid,
        )

        logger.info(
            "Image shape: %s | Pixel spacing: %s",
            expected_shape,
            expected_spacing,
        )

        return DicomSeries(
            study_instance_uid=expected_study_uid,
            series_instance_uid=expected_series_uid,
            slices=slices,
        )
