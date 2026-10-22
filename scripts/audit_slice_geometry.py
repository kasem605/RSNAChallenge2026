
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.preprocessing.dicom.dicom_reader import DicomSeriesReader


def main():
    paths = DatasetPaths.from_root(
        Path(r"D:\RSNA knee abnormality detection 2026")
    )
    metadata = MetadataReader(paths)

    label_columns = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture",
    ]

    train = metadata.train
    labeled = train[train[label_columns].notna().all(axis=1)]
    study_uids = labeled["StudyInstanceUID"].head(3).tolist()

    selector = SeriesSelector(
        train_series_dir=paths.train_series_dir
    )
    reader = DicomSeriesReader()

    for study_uid in study_uids:
        print("\n" + "=" * 72)
        print("Study:", study_uid)

        selection = selector.select(study_uid, metadata.train_series)

        for plane in ("sagittal", "coronal", "axial"):
            selected = getattr(selection, plane, None)
            if selected is None:
                print(f"\n{plane}: no series selected")
                continue

            series = reader.read(selected.series_path)
            ordered = series.ordered_slices()

            print(f"\n{plane.upper()}")
            print("Images:", len(ordered))

            if len(ordered) < 2:
                print("WARNING: insufficient slices for spacing validation")
                continue

            positions = []
            orientations = []

            for slice_ in ordered:
                if slice_.image_position is None:
                    positions = []
                    break
                if slice_.image_orientation is None:
                    orientations = []
                    break

                positions.append(np.asarray(slice_.image_position))
                orientations.append(
                    np.asarray(slice_.image_orientation)
                )

            if len(positions) < 2 or len(orientations) != len(ordered):
                print("WARNING: incomplete spatial geometry")
                continue

            reference = orientations[0]
            orientation_error = max(
                float(np.max(np.abs(item - reference)))
                for item in orientations
            )

            row = reference[:3]
            column = reference[3:]
            normal = np.cross(row, column)
            normal /= np.linalg.norm(normal)

            projected = np.asarray(positions) @ normal
            differences = np.diff(projected)
            absolute_differences = np.abs(differences)
            nonzero = absolute_differences[
                absolute_differences > 1e-5
            ]

            print("First position:", positions[0].tolist())
            print("Last position:", positions[-1].tolist())
            print("Projection direction:", "increasing" if np.all(differences > 0)
                  else "decreasing" if np.all(differences < 0)
                  else "not strictly monotonic")
            print("Orientation max deviation:", orientation_error)
            print("Slice spacing median (mm):",
                  float(np.median(nonzero)) if len(nonzero) else None)
            print("Slice spacing min/max (mm):",
                  (
                      float(np.min(nonzero)),
                      float(np.max(nonzero)),
                  ) if len(nonzero) else None)

            if orientation_error > 1e-3:
                print("WARNING: inconsistent slice orientations")

            if len(nonzero) != len(differences):
                print("WARNING: duplicate or nearly duplicate slice positions")

            if len(nonzero) and (
                np.max(nonzero) - np.min(nonzero)
                > max(0.1, 0.05 * np.median(nonzero))
            ):
                print("WARNING: variable slice spacing")

    print("\nSLICE GEOMETRY AUDIT COMPLETE")


if __name__ == "__main__":
    main()
