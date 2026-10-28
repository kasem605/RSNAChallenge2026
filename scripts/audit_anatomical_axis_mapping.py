
from pathlib import Path
import sys
import itertools

import numpy as np

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.preprocessing.dicom.dicom_reader import DicomSeriesReader


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATASET_ROOT = Path(r"D:\RSNA knee abnormality detection 2026")

STUDY_UIDS = [
    "1.2.826.0.1.3680043.8.498.10095687747295410396510538520594649149",
    "1.2.826.0.1.3680043.8.498.10170898615867673028696505248839028269",
    "1.2.826.0.1.3680043.8.498.10306159113324811538703788080836752052",
]

# A direction within 15 degrees of a patient coordinate axis
# is considered reasonably axis-aligned for this diagnostic.
MAX_AXIS_DEVIATION_DEGREES = 15.0

PATIENT_AXES = ("X (Left/Right)", "Y (Anterior/Posterior)",
                "Z (Inferior/Superior)")


# ------------------------------------------------------------
# Vector helpers
# ------------------------------------------------------------

def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a normalized vector, rejecting invalid directions."""

    length = np.linalg.norm(vector)

    if not np.isfinite(length) or length < 1e-8:
        raise ValueError("Invalid or zero-length direction vector.")

    return vector / length


def direction_report(
    axis_name: str,
    direction: np.ndarray,
) -> dict:
    """
    Find the closest patient-coordinate axis.

    Alignment is the absolute dot product, so it measures how
    closely the direction follows an axis regardless of sign.
    """

    direction = normalize(direction)

    # Patient-coordinate unit vectors:
    # X = left/right, Y = anterior/posterior, Z = inferior/superior.
    patient_basis = np.eye(3)

    dot_products = patient_basis @ direction
    dominant_axis = int(np.argmax(np.abs(dot_products)))

    alignment = float(abs(dot_products[dominant_axis]))
    angle_degrees = float(
        np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0)))
    )

    sign = "+" if dot_products[dominant_axis] >= 0 else "-"

    return {
        "axis_name": axis_name,
        "direction": direction,
        "patient_axis": PATIENT_AXES[dominant_axis],
        "sign": sign,
        "alignment": alignment,
        "angle_degrees": angle_degrees,
        "dominant_axis_index": dominant_axis,
    }


# ------------------------------------------------------------
# Orientation analysis
# ------------------------------------------------------------

def analyze_series(series):
    """
    Map original NumPy axes to patient coordinates.

    Original NumPy volume axes:
        axis 0 = ordered slices
        axis 1 = image rows
        axis 2 = image columns

    DICOM ImageOrientationPatient:
        first triplet  = direction of increasing column index
        second triplet = direction of increasing row index
    """

    ordered_slices = series.ordered_slices()

    if not ordered_slices:
        raise ValueError("Series contains no ordered slices.")

    first_slice = ordered_slices[0]
    orientation = first_slice.image_orientation

    if orientation is None or len(orientation) != 6:
        raise ValueError(
            "A valid six-value ImageOrientationPatient is required."
        )

    # Normalize the DICOM direction cosines.
    column_direction = normalize(
        np.asarray(orientation[:3], dtype=float)
    )
    row_direction = normalize(
        np.asarray(orientation[3:], dtype=float)
    )

    # Verify the two in-plane directions are approximately orthogonal.
    in_plane_dot = float(np.dot(column_direction, row_direction))

    if abs(in_plane_dot) > 1e-3:
        raise ValueError(
            f"In-plane direction vectors are not orthogonal: "
            f"dot product={in_plane_dot:.6f}"
        )

    # The slice normal follows the direction used by the cross
    # product of the DICOM direction cosines.
    slice_normal = normalize(
        np.cross(column_direction, row_direction)
    )

    # DicomSeries.ordered_slices() sorts by increasing projection
    # onto this normal. Therefore NumPy axis 0 follows +normal.
    source_axes = [
        ("NumPy axis 0 (slice/depth)", slice_normal),
        ("NumPy axis 1 (row)", row_direction),
        ("NumPy axis 2 (column)", column_direction),
    ]

    reports = [
        direction_report(name, vector)
        for name, vector in source_axes
    ]

    # Measure orientation consistency across the series.
    max_orientation_deviation = 0.0

    for slice_ in ordered_slices:
        current = slice_.image_orientation

        if current is None or len(current) != 6:
            raise ValueError(
                f"Missing orientation metadata: {slice_.path}"
            )

        current_column = normalize(
            np.asarray(current[:3], dtype=float)
        )
        current_row = normalize(
            np.asarray(current[3:], dtype=float)
        )

        for reference, observed in (
            (column_direction, current_column),
            (row_direction, current_row),
        ):
            dot = np.clip(np.dot(reference, observed), -1.0, 1.0)
            angle = float(np.degrees(np.arccos(dot)))
            max_orientation_deviation = max(
                max_orientation_deviation,
                angle,
            )

    return reports, max_orientation_deviation, in_plane_dot


def print_series_report(
    study_uid: str,
    plane_name: str,
    selected,
    reader: DicomSeriesReader,
):
    print(f"\n--- {plane_name.upper()} ---")

    if selected is None:
        print("No selected series.")
        return

    try:
        series = reader.read(selected.series_path)

        reports, orientation_deviation, in_plane_dot = (
            analyze_series(series)
        )

        print(f"Series UID: {series.series_instance_uid}")
        print(f"Images: {len(series.slices)}")
        print(f"Study UID: {study_uid}")
        print(f"In-plane direction dot product: {in_plane_dot:.8f}")
        print(
            "Maximum orientation deviation across slices: "
            f"{orientation_deviation:.6f} degrees"
        )

        print("\nOriginal NumPy axis mapping:")

        dominant_axes = []

        for report in reports:
            direction_text = np.array2string(
                report["direction"],
                precision=6,
                suppress_small=True,
            )

            print(f"\n  {report['axis_name']}")
            print(f"    Direction vector: {direction_text}")
            print(
                f"    Closest patient axis: "
                f"{report['sign']} {report['patient_axis']}"
            )
            print(
                f"    Alignment score: {report['alignment']:.6f}"
            )
            print(
                f"    Angular deviation: "
                f"{report['angle_degrees']:.3f} degrees"
            )

            dominant_axes.append(
                report["dominant_axis_index"]
            )

            if (
                report["angle_degrees"]
                > MAX_AXIS_DEVIATION_DEGREES
            ):
                print(
                    "    WARNING: Direction exceeds the configured "
                    "axis-alignment tolerance."
                )

        if len(set(dominant_axes)) != 3:
            print(
                "\n  WARNING: Multiple NumPy axes map to the same "
                "dominant patient axis. Review the geometry."
            )

        if all(
            report["angle_degrees"] <= MAX_AXIS_DEVIATION_DEGREES
            for report in reports
        ) and len(set(dominant_axes)) == 3:
            print(
                "\n  Axis-alignment diagnostic: PASS "
                "(within configured tolerance)."
            )
        else:
            print(
                "\n  Axis-alignment diagnostic: REVIEW "
                "(inspect obliquity and coordinate mapping)."
            )

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("=" * 78)
    print("ANATOMICAL AXIS MAPPING AUDIT")
    print("=" * 78)
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Maximum axis deviation: {MAX_AXIS_DEVIATION_DEGREES} degrees")
    print("Read-only audit: no image data will be modified.")

    paths = DatasetPaths.from_root(DATASET_ROOT)
    metadata = MetadataReader(paths)

    selector = SeriesSelector(
        train_series_dir=paths.train_series_dir
    )
    reader = DicomSeriesReader()

    failures = []

    for study_uid in STUDY_UIDS:
        print(f"\n{'=' * 78}")
        print(f"Study: {study_uid}")
        print("=" * 78)

        try:
            selection = selector.select(
                study_uid,
                metadata.train_series,
            )

            for plane_name in ("sagittal", "coronal", "axial"):
                selected = getattr(selection, plane_name, None)

                try:
                    print_series_report(
                        study_uid,
                        plane_name,
                        selected,
                        reader,
                    )
                except Exception as exc:
                    failures.append(
                        (study_uid, plane_name, str(exc))
                    )

        except Exception as exc:
            failures.append(
                (study_uid, "study selection", str(exc))
            )
            print(f"Study failed: {type(exc).__name__}: {exc}")

    print(f"\n{'=' * 78}")
    print("AUDIT SUMMARY")
    print("=" * 78)

    if failures:
        print(f"Failures: {len(failures)}")

        for study_uid, series_name, error in failures:
            print(f"- {study_uid} | {series_name}: {error}")
    else:
        print("No unhandled processing failures.")

    print(
        "\nInterpretation: alignment scores describe direction similarity. "
        "They do not validate left/right labeling, anatomical correctness, "
        "or a complete physical-coordinate transformation."
    )


if __name__ == "__main__":
    main()
