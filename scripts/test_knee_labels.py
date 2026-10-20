from ish_knee.dataset.knee_labels import KneeLabels

labels = KneeLabels(
    acl=1,
    mcl=0,
    medial_meniscus=1,
    lateral_meniscus=0,
    medial_oa=1,
    lateral_oa=0,
    pf_oa=1,
    effusion=0,
    synovitis=1,
    bakers=0,
    contusion=0,
    fracture=0
)

print("=" * 70)
print("KNEE LABELS TEST")
print("=" * 70)

print()
print("Labels:", labels.as_tuple)
print("Label count:", labels.count)

assert labels.count == 12
assert labels.acl == 1
assert labels.mcl == 0
assert labels.fracture == 0

try:
    KneeLabels(
        acl=2,
        mcl=0,
        medial_meniscus=1,
        lateral_meniscus=0,
        medial_oa=1,
        lateral_oa=0,
        pf_oa=1,
        effusion=0,
        synovitis=1,
        bakers=0,
        contusion=0,
        fracture=0
    )
    raise AssertionError("Invalid label value should have raised ValueError")

except ValueError:
    pass

print()
print("KNEE LABELS TEST PASSED")
print("=" * 70)