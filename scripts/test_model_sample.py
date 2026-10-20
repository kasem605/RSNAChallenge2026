import numpy as np

from ish_knee.model.model_input import ModelInput
from ish_knee.model.model_target import ModelTarget
from ish_knee.model.model_sample import ModelSample

def main():

    print("=" * 70)
    print("MODEL SAMPLE TEST")
    print("=" * 70)

    # ------------------------------------------------------------
    # Create synthetic processed MRI volumes
    # ------------------------------------------------------------

    sagittal = np.zeros((8, 64, 64), dtype=np.float32)
    coronal = np.zeros((8, 64, 64), dtype=np.float32)
    axial = np.zeros((8, 64, 64), dtype=np.float32)

    model_input = ModelInput(
        study_instance_uid="TEST-STUDY",
        sagittal=sagittal,
        coronal=coronal,
        axial=axial
    )

    target = ModelTarget(
        values=np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],dtype=np.float32)
    )

    sample = ModelSample(input=model_input, target=target)

    assert  sample.input is model_input
    assert sample.target is target
    assert sample.study_instance_uid == "TEST-STUDY"

    print()
    print("Study UID:       ", sample.study_instance_uid)
    print("Sagittal shape:  ", sample.input.sagittal.shape)
    print("Coronal shape:   ", sample.input.coronal.shape)
    print("Axial shape:     ", sample.input.axial.shape)
    print("Target count:    ", sample.target.count)

    # ------------------------------------------------------------
    # Invalid input test
    # ------------------------------------------------------------

    try:
        ModelSample(input="invalid", target=ModelTarget)

        raise AssertionError("Expected TypeError for invalid ModelInput.")

    except TypeError as error:
        print()
        print("Expected TypeError received")
        print(error)

    # ------------------------------------------------------------
    # Invalid target test
    # ------------------------------------------------------------

    try:
        ModelSample(input=model_input, target="invalid")

        raise AssertionError("Expected TypeError for invalid ModelTarget.")

    except TypeError as error:
        print()
        print("Expected TypeError received")
        print(error)


    print()
    print("MODEL SAMPLE TEST PASSED")
    print("=" * 70)
    
if __name__ == "__main__":
    main()
