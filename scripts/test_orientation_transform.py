import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    PROJECT_ROOT / "src"
)

from ish_knee.preprocessing.volume.orientation_transform import OrientationTransform

print("=" * 70)
print("ORIENTATION TRANSFORM TEST")
print("=" * 70)

# --------------------------------------------------------------------
# Identity transformation
# --------------------------------------------------------------------

identity = OrientationTransform(
    axis_order=(0, 1, 2),
    flip_axis_0=False,
    flip_axis_1=False,
    flip_axis_2=False
)

print()
print("Identity transformation")
print("Axis order:", identity.axis_order)
print("Flip axis 0:", identity.flip_axis_0)
print("Flip axis 1:", identity.flip_axis_1)
print("Flip axis 2:", identity.flip_axis_2)

assert identity.is_identity is True

# --------------------------------------------------------------------
# Transformation with axis permutation
# --------------------------------------------------------------------

permuted =  OrientationTransform(
    axis_order=(2, 1, 0),
    flip_axis_0=False,
    flip_axis_1=False,
    flip_axis_2=False
)

assert permuted.is_identity is False

# --------------------------------------------------------------------
# Transformation with a flip
# --------------------------------------------------------------------

flipped =  OrientationTransform(
    axis_order=(2, 1, 0),
    flip_axis_0=False,
    flip_axis_1=False,
    flip_axis_2=False
)

assert flipped.is_identity is False

print()
print("=" * 70)
print("ORIENTATION TRANSFORM TEST PASSED")
print("=" * 70)