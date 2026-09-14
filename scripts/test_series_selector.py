from pathlib import Path
import sys

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

print(PROJECT_ROOT)

from ish_knee.data import (
    DatasetPaths,
    MetadataReader
)

from ish_knee.preprocessing import (
    SeriesSelector
)

def print_selected (name, selected ):
    print()
    print(name)
    print("=" * 50 )

    if selected is None:
        print("NOT FOUND")
        return

    print(
        f"Series UID: "
        f"{selected.series_instance_uid}"
    )

    print(
        f"Fluid Sensitive: "
        f"{selected.fluid_sensitive}"
    )  

    print(
        f"Fat Suppression: "
        f"{selected.fat_suppression}"
    )

    print(
        f"Series Count: "
        f"{selected.file_count}"
    )

    print(
        f"Path: "
        f"{selected.series_path}"
    )

def main():

    dataset_root = r"D:\RSNA knee abnormality detection 2026"

    paths = DatasetPaths.from_root(
        dataset_root
    )

    paths.validate()

    metadata = MetadataReader(
        paths
    )

    metadata.validate()

    study_uid = (
        metadata.get_study_uids()[0]
    )

    selector = SeriesSelector(
        paths.train_series_dir
    )

    selection = selector.select(
        study_uid,
        metadata.train_series
    )

    print()
    print("=" * 70)
    print("SERIES SELECTION")
    print("=" * 70)

    print(
        f"Study: {study_uid}"
    )

    print_selected(
        "SAGITTAL",
        selection.sagittal
    )

    print_selected(
        "CORONAL",
        selection.coronal
    )

    print_selected(
        "AXIAL",
        selection.axial
    )

if __name__ == "__main__":
    main()