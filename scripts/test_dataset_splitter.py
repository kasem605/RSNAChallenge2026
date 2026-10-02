from ish_knee.dataset.dataset_sample import DatasetSample
from ish_knee.dataset.knee_study_sample import KneeStudySample
from ish_knee.dataset.knee_labels import KneeLabels
from ish_knee.dataset.dataset_splitter import DatasetSplitter

def create_sample(uid: str) -> DatasetSample:

    study = KneeStudySample(
        study_instance_uid=uid,
        sagittal=None,
        coronal=None,
        axial=None
    )

    labels = KneeLabels(
        acl=0,
        mcl=0,
        medial_meniscus=0,
        lateral_meniscus=0,
        medial_oa=0,
        lateral_oa=0,
        pf_oa=0,
        effusion=0,
        synovitis=0,
        bakers=0,
        contusion=0,
        fracture=0
    )

    return DatasetSample(study=study, labels=labels)

samples = [
    create_sample(f"STUDY-{index:03d}")
    for index in range(100)
]

splitter = DatasetSplitter()

split = splitter.split(
    samples=samples,
    validation_fraction=0.20,
    random_seed=42
)

print()
print("=" * 70)
print("DATASET SPLITTER TEST")
print("=" * 70)

print()
print("Total samples:", split.total_count)
print("Training samples:", split.train_count)
print("Validation_samples:", split.validation_count)

assert split.total_count == 100
assert split.train_count == 80
assert split.validation_count == 20

train_uids = {
    sample.study_instance_uid
    for sample in split.train_samples
}

validation_uids = {
    sample.study_instance_uid
    for sample in split.validation_samples
}

assert train_uids.isdisjoint(validation_uids)

assert (len(train_uids) == 80)

assert (len(validation_uids) == 20)

# verify reproducability

split_again = splitter.split(
    samples=samples,
    validation_fraction=0.20,
    random_seed=42
)

assert [
    sample.study_instance_uid
    for sample in split.train_samples
] == [
    sample.study_instance_uid
    for sample in split_again.train_samples    
]

assert [
    sample.study_instance_uid
    for sample in split.validation_samples
] == [
    sample.study_instance_uid
    for sample in split_again.validation_samples    
]

print()
print("Study leakage check: PASSED")
print("Reproducability check: PASSED")

print()
print("DATASET SPLITTER TEST PASSED")
print("=" * 70)