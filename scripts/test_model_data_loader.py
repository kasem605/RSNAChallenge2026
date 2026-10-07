from torch.utils.data import DataLoader

from ish_knee.model.model_data_loader import MOdelDataLoader

def main():

    print("=" * 70)
    print("MODEL DATA LOADER IMPORT TEST")
    print("=" * 70)

    assert callable(MOdelDataLoader)
    assert DataLoader is not None

    print("ModelDataLoader import successful")
    print("Pytorch DataLoader import successful")
    print("MODEL DATA LOADER IMPORT TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()

