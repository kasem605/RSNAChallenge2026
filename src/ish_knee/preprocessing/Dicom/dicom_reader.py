from pathlib import Path

import numpy as np
import pydicom
from .dicom_series import DicomSeries
from .dicomslice import DicomSlice

class DicomSeriesReader:
    """
    Reads a DICOM series from a directory
    """

    def read(self, series_path: str | Path) -> DicomSeries:

        series_path = Path(series_path)

        if not series_path.is_dir():
            raise FileNotFoundError(
                f"DICOM series directory not found: "
                f"{series_path}"
            )

        slices: list[DicomSlice] = []

        for file_path in series_path.iterdir():

            if not file_path.is_file():
                continue

            try:
                dataset = pydicom.dcmread(
                    file_path,
                    force=True
                )

                pixel_array = dataset.pixel_array

            except Exception:
                continue

            instance_number = int(
                getattr(
                    dataset,
                    "InstanceNumber",
                    0
                )
            )

            image_position = self._get_image_position(
                dataset
            )

            slices.append(
                DicomSlice(
                    path=file_path,
                    instance_number=instance_number,
                    image_position=image_position,
                    pixel_array=pixel_array
                )
            )

            if not slices:
                raise ValueError(
                    f"No readable DICOM images found in "
                    f"{series_path}"
                )

            slices.sort(key=lambda item: item.instance_number)

            return DicomSeries(
                series_path=series_path,
                slices=tuple(slices)
            )

    @staticmethod
    def _get_image_position( dataset) -> tuple[float, float, float] | None:

        position = getattr(
            dataset,
            "ImagePositionPatient",
            None
        )

        if position is None:
            return None

        return(
            float(position[0]),
            float(position[1])
        )

        


        