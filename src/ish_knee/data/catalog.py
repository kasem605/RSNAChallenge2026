from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .metadata import MetadataReader
from .paths import DatasetPaths

class SeriesCatalogBuilder:
    """
    Builds a catalog of all MRI series in the
    RSNA Knee dataset

    This class does not read DICOM pixel data.
    It combines train_series.csv metadata with filesystem
    information
    """

    def __init__(self, paths: DatasetPaths, metadata: MetadataReader):

        self._paths = paths
        self._metadata = metadata

    def build(self, limit: int | None = None)-> pd.DataFrame:
        """
        Build the series catalog

        Parameters
        ----------
        limit:
            Optional maximum number of series to process,
            Useful for testing.
        """

        train_series = ( self._metadata.train_series.copy() )

        if limit is not None:
            train_series = train_series.head(limit)

        records=[]

        for row in tqdm(
            train_series.itertuples(index=False),
            total=len(train_series),
            desc="Building series catalog",
        ):
            study_uid =str(row.StudyInstanceUID)

            series_uid = str(row.SeriesInstanceUID)

            series_path = (self._paths.series_dir(study_uid, series_uid))

            file_count = self._count_dicom_files(series_path)

            record = {
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "Anatomical_PLane": row.Anatomical_Plane,
                "Fluid_Sensitive": row.Fluid_Sensitive,
                "Fat_Suppression": row.Fat_Suppression,
                "SeriesPath": str(series_path),
                "FileCount": file_count,
                "SeriesExists": series_path.exists(),
            }

            records.append(record)

        return pd.DataFrame(records)

    @staticmethod
    def _count_dicom_files(series_path: Path,)->int:

        """
        Count DICOM files in a series directory.

        We count all files because DICOM datasets may
        not consistently use the .dcm extension
        """

        if not series_path.exists():
            return 0

        return sum(
            1
            for path in series_path.iterdir()
            if path.is_file()
        )

