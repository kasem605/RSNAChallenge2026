
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    manifest_path = (
        project_root
        / "outputs"
        / "processed_dataset_v1"
        / "manifest.csv"
    )

    dataset = KneeMRIDataset(
        manifest_path=manifest_path,
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    print("\n" + "=" * 72)
    print("CNN DATASET INPUT DIAGNOSTIC")
    print("=" * 72)

    study_stats = []

    for study_uids, images, labels in loader:
        print(f"\nBatch image shape: {tuple(images.shape)}")
        print(f"Batch label shape: {tuple(labels.shape)}")

        for i, study_uid in enumerate(study_uids):
            image = images[i].float()
            label = labels[i]

            stats = {
                "uid": study_uid,
                "mean": image.mean().item(),
                "std": image.std().item(),
                "min": image.min().item(),
                "max": image.max().item(),
            }
            study_stats.append(stats)

            plane_stats = []
            for channel, plane in enumerate(
                ("sagittal", "coronal", "axial")
            ):
                volume = image[channel]
                plane_stats.append(
                    f"{plane}: mean={volume.mean().item():.4f}, "
                    f"std={volume.std().item():.4f}"
                )

            print(f"\nStudy: {study_uid}")
            print(
                f"  Overall: mean={stats['mean']:.5f}, "
                f"std={stats['std']:.5f}, "
                f"min={stats['min']:.5f}, "
                f"max={stats['max']:.5f}"
            )
            print(f"  Labels: {int(label.sum().item())} positive")
            for result in plane_stats:
                print(f"  {result}")

        if len(study_stats) >= 12:
            break

    if len(study_stats) >= 2:
        means = [item["mean"] for item in study_stats]
        stds = [item["std"] for item in study_stats]

        print("\n" + "=" * 72)
        print("SUMMARY ACROSS STUDIES")
        print("=" * 72)
        print(f"Studies checked: {len(study_stats)}")
        print(f"Mean range: {min(means):.6f} to {max(means):.6f}")
        print(f"Std range:  {min(stds):.6f} to {max(stds):.6f}")


if __name__ == "__main__":
    main()
