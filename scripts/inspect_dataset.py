from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from ish_knee.data import (
    DatasetPaths,
    MetadataReader
)

def main():

    dataset_root=r"D:\RSNA knee abnormality detection 2026"

    paths = DatasetPaths.from_root(
        dataset_root
    )

    paths.validate()

    metadata = MetadataReader(paths)

    metadata.validate()

    metadata.print_summary()
 
    studies = metadata.get_study_uids()

    print()

    print(len(studies))
    print()
    print(studies[:5])


if __name__ == "__main__":
    main()