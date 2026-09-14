from pathlib import Path
import pandas as pd

class MetadataReader:
    """
    Loads and validates RSNA KNee dataset metadata
    This class does not read DICOM pixel data.
    """

    def __init__(self, paths):
        self._paths = paths

        self._train: pd.DataFrame | None = None
        self._train_series: pd.DataFrame | None = None

    @property
    def train(self) -> pd.DataFrame:
        """
        Training study metadata
        """

        if(self._train is None):
            self._train = pd.read_csv(
                self._paths.train_csv
            )

        return self._train

    @property
    def train_series(self) -> pd.DataFrame:
        """
        Training MRI series metadata
        """

        if(self._train_series is None):
            self._train_series=pd.read_csv(
                self._paths.train_series_csv
            )

        return self._train_series

    def validate(self) -> None:
        """
        Validate the basic structure of the metadata
        """

        self._validate_train()
        self._validate_train_series()
        self._validate_relationship()


    def _validate_train(self) -> None:

        required_columns = {
            "StudyInstanceUID",
            "Report"
        }

        missing = (
            required_columns - set(self.train.columns)
        )

        if missing:
            raise ValueError(
                "train.csv is missing columns: "
                f"{sorted(missing)}"
                )



    def _validate_train_series(self) -> None:

        required_columns = {
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Fluid_Sensitive",
            "Fat_Suppression",
            "Anatomical_Plane"
        }

        missing = (
            required_columns - set(self.train_series.columns)
        )

        if missing:
            raise ValueError(
                "train_series.csv is missing columns: "
                f"{sorted(missing)}"
                )

    def _validate_relationship(self) -> None:

        train_studies = set(
            self.train["StudyInstanceUID"]
        )

        series_studies = set(
            self.train_series["StudyInstanceUID"]
        )

        orphan_series = set(
            series_studies - train_studies
        )

        if orphan_series:

            raise ValueError(
                f"Found {len(orphan_series):,}"
                "series belonging to studies "
                " that do not exist in train.csv"
            )

    def print_summary(self) -> None:

        train = self.train
        series = self.train_series

        print()
        print("=" * 70)
        print("RSNA KNEE METADATA")
        print("=" * 70)

        print(
            f"Training studies : "
            f"{train['StudyInstanceUID'].nunique():,}"
        )

        print(
            f"Training series : "
            f"{series['SeriesInstanceUID'].nunique():,}"
        )

        print()

        print("train.csv columns:")
        for column in train.columns:
            print(f"    {column}")

        print()

        print("train_series.csv columns:")
        for column in series.columns:
            print(f"    {column}")

    def get_study_series(self, study_id: str) -> pd.DataFrame:
        
        """
        Return all MRI series belonging to a study
        """

        return self.train_series[
            self.train_series["StudyInstanceUID"]
            == study_id
        ].copy()

    def get_study_uids(self) -> list[str]:

        """
        Return all training study IDs.
        """

        return (
            self.train["StudyInstanceUID"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        