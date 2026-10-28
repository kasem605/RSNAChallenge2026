import numpy as np

from ish_knee.dataset.knee_mri_sample import KneeMRISample

sagittal = np.zeros((64,256,256),dtype=np.float32)

coronal = np.zeros((64,256,256),dtype=np.float32)

axial = np.zeros((64,256,256),dtype=np.float32)

sample = KneeMRISample(
    study_instance_uid="TEST-STUDY",
    sagittal=sagittal,
    coronal=coronal,
    axial=axial
)

print("=" * 70)
print("KNEE MRI SAMPLE TEST")
print("=" * 70)

print()
print("StudUID", sample.study_instance_uid)

print()
print("Sagittal shape:", sample.sagittal_shape)
print("Coronal shape:", sample.coronal_shape)
print("axial shape:", sample.axial_shape)

print()
print("All volumes 3-D", sample.is_3d)

assert sample.study_instance_uid == "TEST-STUDY"

assert sample.sagittal.dtype == np.float32
assert sample.coronal.dtype == np.float32
assert sample.axial.dtype == np.float32

print()
print("KNEE MRI SAMPLE TEST PASSED")
print("=" * 70)
