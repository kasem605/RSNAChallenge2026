
from pathlib import Path

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "processed_dataset_v1"
    / "manifest.csv"
)


def main():
    dataset = KneeMRIDataset(
        manifest_path=MANIFEST_PATH,
        split="validation",
    )

    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    study_uid, images, labels = dataset[0]

    print("=" * 60)
    print("DATASET INTERFACE TEST")
    print("=" * 60)

    print(f"Validation studies: {len(dataset)}")
    print(f"Study UID:          {study_uid}")
    print(f"Images shape:       {tuple(images.shape)}")
    print(f"Images dtype:       {images.dtype}")
    print(f"Labels shape:       {tuple(labels.shape)}")
    print(f"Labels dtype:       {labels.dtype}")

    assert isinstance(study_uid, str) and study_uid.strip()
    assert images.shape == (3, 128, 128, 128)
    assert labels.shape == (12,)

    print("\nDATASET INTERFACE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
