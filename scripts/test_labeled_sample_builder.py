
from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader
from ish_knee.data.labeled_sample_builder import LabeledSampleBuilder
from ish_knee.preprocessing import SeriesSelector
from ish_knee.model.study_split import StudySplit


DATASET_ROOT = r"D:\RSNA knee abnormality detection 2026"


def main():
    paths = DatasetPaths.from_root(DATASET_ROOT)
    metadata = MetadataReader(paths)

    selector = SeriesSelector(paths.train_series_dir)
    builder = LabeledSampleBuilder(metadata, selector)

    samples = builder.build()

    complete = [sample for sample in samples if sample.study.is_complete]
    incomplete = [sample for sample in samples if not sample.study.is_complete]

    print("=" * 70)
    print("LABELED SAMPLE BUILDER TEST")
    print("=" * 70)
    print(f"Fully labeled samples: {len(samples)}")
    print(f"Complete three-plane studies: {len(complete)}")
    print(f"Incomplete three-plane studies: {len(incomplete)}")

    if not samples:
        raise AssertionError("No labeled samples were built.")

    for sample in samples[:3]:
        print(f"\nStudy UID: {sample.study_instance_uid}")
        print(f"Labels: {sample.labels.as_tuple}")
        print(f"Planes available: {sample.study.plane_count}")

    # Split by study, not by MRI series or individual slices.
    training, validation = StudySplit.split(
        samples,
        validation_fraction=0.30,
        seed=42,
    )

    train_uids = {sample.study_instance_uid for sample in training}
    validation_uids = {
        sample.study_instance_uid for sample in validation
    }

    assert train_uids.isdisjoint(validation_uids)
    assert len(training) + len(validation) == len(samples)

    print("\nStudy split:")
    print(f"Training studies: {len(training)}")
    print(f"Validation studies: {len(validation)}")
    print(f"UID overlap: {len(train_uids & validation_uids)}")

    print("\nLABELED SAMPLE BUILDER TEST PASSED")


if __name__ == "__main__":
    main()
