
from pathlib import Path
import sys
import traceback

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")

LABELED_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "labeled_study_audit"
    / "fully_labeled_studies.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_manifest"

LABEL_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

PLANES = ("sagittal", "coronal", "axial")


def add_project_root_to_path():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def selected_series_details(selection):
    """Extract a series UID and path from a SeriesSelector result."""
    if selection is None:
        return None, None

    # Accommodate common attribute names without assuming one
    # particular SelectedSeries implementation.
    uid = None
    for name in ("series_instance_uid", "SeriesInstanceUID"):
        if hasattr(selection, name):
            uid = getattr(selection, name)
            break

    path = None
    for name in ("series_path", "path", "directory"):
        if hasattr(selection, name):
            path = getattr(selection, name)
            break

    if uid is not None:
        uid = str(uid)

    if path is not None:
        path = str(path)

    return uid, path


def main():
    add_project_root_to_path()

    from ish_knee.data.paths import DatasetPaths
    from ish_knee.data.metadata import MetadataReader
    from ish_knee.preprocessing.series_selector import SeriesSelector

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not LABELED_CSV.is_file():
        raise FileNotFoundError(
            f"Labeled study CSV not found: {LABELED_CSV}"
        )

    labeled = pd.read_csv(LABELED_CSV)

    required_columns = ["StudyInstanceUID", *LABEL_COLUMNS]
    missing = [
        col for col in required_columns
        if col not in labeled.columns
    ]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    if labeled["StudyInstanceUID"].duplicated().any():
        raise ValueError("Duplicate StudyInstanceUID values found.")

    paths = DatasetPaths.from_root(DATA_ROOT)
    metadata = MetadataReader(paths)
    metadata.validate()

    selector = SeriesSelector(
        train_series_dir=paths.train_series_dir
    )

    manifest_rows = []
    failure_rows = []

    for index, row in labeled.iterrows():
        study_uid = str(row["StudyInstanceUID"])

        print(
            f"[{index + 1}/{len(labeled)}] "
            f"Checking study {study_uid}"
        )

        try:
            selected = selector.select(
                study_uid,
                metadata.train_series,
            )

            record = {
                "StudyInstanceUID": study_uid,
                **{label: row[label] for label in LABEL_COLUMNS},
            }

            missing_planes = []

            for plane in PLANES:
                selection = getattr(selected, plane, None)
                series_uid, series_path = selected_series_details(
                    selection
                )

                record[f"{plane}_SeriesInstanceUID"] = series_uid
                record[f"{plane}_Path"] = series_path

                if selection is None:
                    missing_planes.append(plane)
                elif not series_uid or not series_path:
                    missing_planes.append(plane)

            record["AllPlanesAvailable"] = not missing_planes
            record["MissingPlanes"] = ",".join(missing_planes)

            manifest_rows.append(record)

            if missing_planes:
                failure_rows.append({
                    "StudyInstanceUID": study_uid,
                    "FailureType": "MissingOrIncompleteSelection",
                    "Details": ",".join(missing_planes),
                })

        except Exception as exc:
            failure_rows.append({
                "StudyInstanceUID": study_uid,
                "FailureType": type(exc).__name__,
                "Details": str(exc),
                "Traceback": traceback.format_exc(),
            })

    manifest_path = OUTPUT_DIR / "study_series_manifest.csv"
    failures_path = OUTPUT_DIR / "series_selection_failures.csv"

    pd.DataFrame(manifest_rows).to_csv(
        manifest_path, index=False
    )
    pd.DataFrame(failure_rows).to_csv(
        failures_path, index=False
    )

    if manifest_rows:
        manifest = pd.DataFrame(manifest_rows)
        complete_count = int(manifest["AllPlanesAvailable"].sum())
    else:
        complete_count = 0

    print()
    print("=" * 70)
    print("SERIES AVAILABILITY AUDIT")
    print("=" * 70)
    print(f"Labeled studies: {len(labeled)}")
    print(f"Studies processed by selector: {len(manifest_rows)}")
    print(f"Studies with all three selections: {complete_count}")
    print(f"Studies with selection issues: {len(failure_rows)}")
    print(f"Manifest: {manifest_path}")
    print(f"Failure report: {failures_path}")


if __name__ == "__main__":
    main()
