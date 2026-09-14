from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DatasetPaths:

    """
    Defines the local filesystem locations for the
    RSNA Knee Abnormality Detection Path
    """
    root: Path
    train_csv: Path
    train_series_csv: Path
    train_series_dir: Path

    test_csv: Path
    test_series_csv: Path
    test_series_dir: Path

    sample_submission_csv: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "DatasetPaths":

        """
        Create dataset paths from the root directory
        """
        root = Path(root).expanduser().resolve()
        
        return cls(
            root=root,
            train_csv=root/"train.csv",
            train_series_csv=root/"train_series.csv",
            train_series_dir=root/"train_series",
            test_csv=root/"test.csv",
            test_series_csv=root/"test_series.csv",
            test_series_dir=root/"test_series",

            sample_submission_csv=(
                root/"sample_submission.csv"
            ),
        )

    def validate(self) -> None:
        """
        Validate that the expected dataset structure exists
        """

        required_paths = {
            "root": self.root,
            "train.csv": self.train_csv,
            "train_series.csv": self.train_series_csv,
            "train_series": self.train_series_dir,
        }

        missing=[]

        for name, path in required_paths.items():

            if not path.exists():
                missing.append( f"{name}: {path}")

        if missing:

            message = (
                "RSNA Knee dataset validation failed.\n\n"
                "Missing paths:\n"
                + "\n".join(
                    f" - {item}"
                    for item in missing
                )
            )

            raise FileNotFoundError(message)

    def print_summary(self) -> None:

        """
        Print the configured dataset paths and whether
        each path exists.
        """

        print()
        print("=" * 70)
        print("RSNA KNEE DATASET")
        print("=" * 70)

        paths = {
            "Root": self.root,
            "Train CSV": self.train_csv,
            "Train Series CSV": self.train_series_csv,
            "Train Series Directory": self.train_series_dir,
            "Test CSV": self.test_csv,
            "Test Series CSV": self.test_series_csv,
            "Test Series Directory": self.test_series_dir,
            "Sample Submission": self.sample_submission_csv,
        }

        for name, path in paths.items():

            status= "OK"  if path.exists()  else "MISSING"

            print(
                f"{name:<25} "
                f"[{status}] "
                f"{path}"
            )

        print()

    def study_dir(self, study_uid: str)-> Path:
        """
        Return the directory containing all series
        for a training study
        """

        return (
            self.train_series_dir
            / str(study_uid)
        )

    def series_dir(self, study_uid: str, series_uid: str) -> Path:
        """
        Return the directory containing the DICOM files
        for a specific training series
        """

        return (
            self.study_dir(study_uid)
            / str(series_uid)
        )