from ish_knee.dataset.dataset_split import DatasetSplit

train_samples =()
validation_samples=()

split = DatasetSplit(train_samples=train_samples, validation_samples=validation_samples)

print("=" * 70)
print("DATSPLIT SPLIT TEST")
print("=" * 70)

print()
print("Training samples:", split.train_count)

print("Validation samples:", split.validation_count)
print("Total samples:", split.total_count)

assert split.train_count == 0
assert split.validation_count == 0
assert split.total_count == 0

print()
print("DATASET SPLIT TEST PASSED")
print("+" * 70)

