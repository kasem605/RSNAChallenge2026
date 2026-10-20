
from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader

DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"


def main():
    paths = DatasetPaths.from_root(DATASET_ROOT)
    metadata = MetadataReader(paths)

    train = metadata.train

    print("=" * 70)
    print("TRAINING LABEL COLUMN INSPECTION")
    print("=" * 70)

    print("\nAll columns:")
    for column in train.columns:
        print(f"  {column}")

    print("\nColumns containing diagnostic terms:")
    keywords = (
        "acl",
        "mcl",
        "meniscus",
        "oa",
        "effusion",
        "synovitis",
        "baker",
        "contusion",
        "fracture",
    )

    for column in train.columns:
        if any(word in column.lower() for word in keywords):
            print(f"  {column}")

    print("\nNon-null counts for candidate label columns:")
    candidates = [
        column for column in train.columns
        if any(word in column.lower() for word in keywords)
    ]

    if candidates:
        print(train[candidates].notna().sum().to_string())
    else:
        print("No candidate label columns found.")


if __name__ == "__main__":
    main()
