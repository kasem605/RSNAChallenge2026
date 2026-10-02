from ish_knee.dataset.knee_study_sample import KneeStudySample

sample = KneeStudySample(
    study_instance_uid="TEST-STUDY",
    sagittal=None,
    coronal=None,
    axial=None
)

print("=" * 70)
print("KNEE STUDY SAMPLE TEST")
print("=" * 70)

print()
print("Study UID:", sample.study_instance_uid)
print("Plane count:", sample.plane_count)

print("Has sagittal:", sample.has_sagittal)
print("Has coronal:", sample.has_coronal)
print("Has axial:", sample.has_axial)

print("Complete:", sample.is_complete)

assert sample.study_instance_uid == "TEST-STUDY"
assert sample.plane_count == 0
assert not sample.has_sagittal
assert not sample.has_coronal
assert not sample.has_axial
assert not sample.is_complete

print()
print("KNEE STUDY SAMPLE TEST PASSED")
print("=" * 70)