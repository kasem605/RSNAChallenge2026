from ish_knee.dataset.knee_labels import KneeLabels
from ish_knee.model.model_target import ModelTarget

import numpy as np

print("=" * 70)
print("MODEL TARGET TEST")
print("=" * 70)

# ------------------------------------------------------------
# TEST 1: Create kneelabels
# ------------------------------------------------------------

labels = KneeLabels(
        acl=0,
        mcl=0,
        medial_meniscus=0,
        lateral_meniscus=0,
        medial_oa=0,
        lateral_oa=0,
        pf_oa=1,
        effusion=1,
        synovitis=0,
        bakers=0,
        contusion=0,
        fracture=0
    )

print()
print("TEST 1: KNEE LABELS")

print()
print(labels.as_tuple)

# ------------------------------------------------------------
# TEST 2: Create ModelTarget from KneeLabels
# ------------------------------------------------------------

target = ModelTarget.from_labels(labels)

print()
print("TEST 2: MODEL TARGET")

print("Target values:")
print(target.values)

print("target count:")
print(target.count)

# ------------------------------------------------------------
# Verify values
# ------------------------------------------------------------

expected = np.array(
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], 
    dtype=np.float32
    )

assert np.array_equal(target.values,expected)

assert target.count == 12

print("ModelTarget values verified")

# ------------------------------------------------------------
# TEST 3: Invalid number of values
# ------------------------------------------------------------

print()
print("TEST 3: INVALID TARGET SIZE")

try:

    ModelTarget(
        values=np.zeros(
            11,
            dtype=np.float32
        )
    )

    raise AssertionError("Expected ValueError for incorrect target size.")

except ValueError as error:
    print("Expected ValueError received")
    print(error)

# ------------------------------------------------------------
# TEST 4: Invalid label value
# ------------------------------------------------------------

print()
print("TEST 4: INVALID LABEL VALUE")

try:

    ModelTarget(
        values=np.array(
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2], 
            dtype=np.float32
        )
    )

    raise AssertionError("Expected ValueError for invalid label value.")

except ValueError as error:
    print("Expected ValueError received")
    print(error)

# ------------------------------------------------------------
# TEST 5: Invalid dimensionality
# ------------------------------------------------------------

print()
print("TEST 4: INVALID TARGET DIMENSION")

try:

    ModelTarget(
        values=np.zeros(
            (12,1), 
            dtype=np.float32
        )
    )

    raise AssertionError("Expected ValueError for 2-D target.")

except ValueError as error:
    print("Expected ValueError received")
    print(error)

print()
print("=" * 70)
print("MODEL TARGET TEST PASSED")
print("=" * 70)
