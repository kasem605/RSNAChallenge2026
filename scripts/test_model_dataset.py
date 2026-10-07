from ish_knee.model.model_dataset import ModelDataset
from ish_knee.model.model_sample_builder import ModelSampleBuilder

def main():

    print("=" * 70)
    print("MODEL DATASET TEST")
    print("=" * 70)

    # ---------------------------------------------------------------
    # we only need to verify the dataset container here.
    # MRI processing will be tested separately
    # ---------------------------------------------------------------

    class DummyBuilder:
        pass

    # ---------------------------------------------------------------
    # Invalid builder test
    # ---------------------------------------------------------------

    try:

        ModelDataset([], DummyBuilder())
        raise AssertionError("Expected TypeError for invalid sample builder")
    
    except TypeError as error:
        print()
        print("Expected TypeError received")
        print(error)

    print()
    print("MODEL DATASET STRUCTURE TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()