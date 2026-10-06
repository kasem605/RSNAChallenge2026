import numpy as np

from ish_knee.dataset.knee_mri_sample import KneeMRISample
from ish_knee.model.model_input import ModelInput

def main() -> None:

    print("=" * 70)
    print("MODEL INPUT TEST")
    print("=" * 70)

    # ------------------------------------------------------------
    # Create synthetic processed MRI volumes
    # ------------------------------------------------------------

    study_uid = "TEST-STUDY"

    sagittal = np.zeros((8, 64, 64), dtype=np.float32)
    coronal = np.zeros((8, 64, 64), dtype=np.float32)
    axial = np.zeros((8, 64, 64), dtype=np.float32)

    # ------------------------------------------------------------
    # Test 1: Create KneeMRISample
    # ------------------------------------------------------------

    mri_sample = KneeMRISample(
        study_instance_uid=study_uid,
        sagittal=sagittal,
        coronal=coronal,
        axial=axial       
    )

    print()
    print("TEST 1: KNEE MRI SAMPLE")

    print("Study UID:")
    print(mri_sample.study_instance_uid)

    print("Sagittal shape:")
    print(mri_sample.sagittal.shape)

    print("Coronal shape:")
    print(mri_sample.coronal.shape)

    print("Axial shape:")
    print(mri_sample.axial.shape)

    if not mri_sample.is_3d:
        raise AssertionError("KneeMRISample should contain three 3-D volumes")

    print("KneeMRISample created successfully")

    # ------------------------------------------------------------
    # Test 2: Convert KneeMRISample to ModelInput
    # ------------------------------------------------------------  

    model_input = ModelInput.from_mri_sample(mri_sample)

    print()
    print("TEST 2: MODEL INPUT")

    print("Study UID:")
    print(model_input.study_instance_uid)

    print("Sagittal shape:")
    print(model_input.sagittal.shape)

    print("Coronal shape:")
    print(model_input.coronal.shape)

    print("Axial shape:")
    print(model_input.axial.shape)    

    # ------------------------------------------------------------
    # Verify values and metadata
    # ------------------------------------------------------------ 

    assert model_input.study_instance_uid == study_uid, "Study UID mismatch"

    assert model_input.sagittal.shape == (8, 64, 64), "Sagittal shape mismatch"
    assert model_input.coronal.shape == (8, 64, 64), "Coronal shape mismatch"
    assert model_input.axial.shape == (8, 64, 64), "Axial shape mismatch"

    assert np.array_equal(model_input.sagittal, sagittal), "Sagittal volume mismatch"
    assert np.array_equal(model_input.coronal, coronal), "Coronal volume mismatch"  
    assert np.array_equal(model_input.axial, axial), "Axial volume mismatch"

    # ------------------------------------------------------------
    # Test 3: Invalid dimensionality
    # ------------------------------------------------------------ 

    print()
    print("TEST 3: INVALID 2-D VOLUME")

    try:
        ModelInput(
            study_instance_uid=study_uid,
            sagittal=np.zeros((64, 64), dtype=np.float32),  # Invalid 2-D volume
            coronal=coronal,
            axial=axial
        )
        raise AssertionError("Expected ValueError for 2-D  sagittal volume")

    except ValueError as error:
        print(error)

    # ------------------------------------------------------------
    # Test 4: Invalid input to from_mri_sample
    # ------------------------------------------------------------ 

    print()
    print("TEST 4: INVALID MRI SAMPLE")

    try:
        ModelInput.from_mri_sample("not a KneeMRISample")


        raise AssertionError("Expected ValueError for invalid MRI sample")

    except TypeError as error:
        print("Expected TypeError received")
        print(error)    

    print()
    print("=" * 70)
    print("MODEL INPUT TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()