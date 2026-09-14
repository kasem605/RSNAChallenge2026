from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import pandas as pd

@dataclass(frozen=True)
class SelectedSeries:
    """
    Represents the preferred MRI series for one
    anatomical plane
    """

    study_instance_uid: str
    series_instance_uid: str

    anatomical_plane: str

    fluid_sensitive: bool
    fat_suppression: bool

    file_count: int

    series_path: str


@dataclass(frozen=True)
class StudySeriesSelection:

    """
    Contains the selected MRI series for a study
    """

    study_instance_uid: str

    sagittal: Optional[SelectedSeries]
    coronal: Optional[SelectedSeries]
    axial: Optional[SelectedSeries]

class SeriesSelector:
    """
    Selects the preferred MRI series for 
    each anatomical plane
    """

    PLANE_ORDER = (
        "sagittal",
        "coronal",
        "axial"
    )

    def __init__(self, train_series_dir: Path):
        self.train_series_dir = train_series_dir

    def _get_series_path(self, study_uid: str, series_uid: str) -> Path:
        return(
            self.train_series_dir
            / str(study_uid)
            / str(series_uid)
        )

    def _get_file_count(self, series_path: Path) -> int:
        if not series_path.exists():
            return 0

        return sum(
            1
            for file in series_path.iterdir()
            if file.is_file()
        )

    def select(self, study_uid: str, series: pd.DataFrame) -> StudySeriesSelection:

        study_series = series[
            series["StudyInstanceUID"] == study_uid
        ].copy()

        return StudySeriesSelection(

            study_instance_uid=study_uid,

            sagittal=self._select_plane(
                study_series,
                "sagittal",
            ),

            coronal=self._select_plane(
                study_series,
                "coronal",
            ),

            axial=self._select_plane(
                study_series,
                "axial",
            )
        )

    def _select_plane( self, series: pd.DataFrame, plane: str )-> Optional[SelectedSeries]:
        candidates = series[
            series["Anatomical_Plane"]
            .astype(str)
            .str.lower()
            == plane
        ].copy()

        candidates["_series_path"] =  candidates.apply(
            lambda row: str (
                self._get_series_path(
                    str(row["StudyInstanceUID"]),
                    str(row["SeriesInstanceUID"])
                )
            ),
            axis = 1
        )

        candidates["_file_count"] =  (
            candidates["_series_path"]
            .apply(
                lambda path: self._get_file_count(
                    Path(path)
                )
            )
        )

        candidates = self._score_candidates(
            candidates
        )

        best = candidates.iloc[0]
        
        return SelectedSeries(

            study_instance_uid=str(
                best["StudyInstanceUID"]
            ),

            series_instance_uid=str(
                best["SeriesInstanceUID"]
            ),

            anatomical_plane=plane,

            fluid_sensitive=self._to_bool(
                best["Fluid_Sensitive"]
            ),

            fat_suppression=self._to_bool(
                best["Fat_Suppression"]
            ),

            file_count=int(
                best["_file_count"]
            ),
            
            series_path=str(
                best["_series_path"]
            )       
        )

    def _get_series_path(self, study_uid: str, series_uid: str) -> Path:
        return (
            self.train_series_dir
            / str(study_uid)
            / str(series_uid)
        )

    @staticmethod
    def _get_file_count(series_path: Path) -> int:
        if not series_path.is_dir():
            return 0

        return sum(
            1
            for file in series_path.iterdir()
            if file.is_file()
        )

    @staticmethod
    def _score_candidates(candidates: pd.DataFrame) -> pd.DataFrame:

        candidates=candidates.copy()

        candidates["_fluid_score"] = (
            candidates["Fluid_Sensitive"]
            .apply( SeriesSelector._to_bool)
            .astype(int)
        )

        candidates["_fat_score"] = (
            candidates["Fat_Suppression"]
            .apply( SeriesSelector._to_bool)
            .astype(int)
        )
    
        candidates["_file_count_score"] = (
            candidates["_file_count"]
        )

        candidates = candidates.sort_values(
            by=[
                "_fluid_score",
                "_fat_score",
                "_file_count_score"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )

        return candidates

    @staticmethod
    def _to_bool(value) -> bool:

        if isinstance(value, bool):
            return value

        if pd.isna(value):
            return False

        if isinstance(value, str):

            return value.strip().lower() in {
                "true",
                "1",
                "yes",
                "y"
            }

        return bool(value)


