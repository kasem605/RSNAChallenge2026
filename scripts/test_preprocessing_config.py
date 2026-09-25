from ish_knee.preprocessing.volume.presprocessing_config import ProcessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing

def main():

    print()
    print("=" * 70)
    print("PRESPROCESSING CONFIG TEST")
    print("=" * 70)

    # --------------------------------------------------------------
    # Create target spacing
    # --------------------------------------------------------------

    spacing = VoxelSpacing(
        spacing_axis_0=1.0,
        spacing_axis_1=1.0,
        spacing_axis_2=1.0,
    )

    # --------------------------------------------------------------
    # Create configuration
    # --------------------------------------------------------------

    config = ProcessingConfig(
        target_spacing=spacing,
        target_shape=(64, 256, 256)
    )

    print()
    print("Configuration:")
    print(config)

    # --------------------------------------------------------------
    # Verify target spacing
    # --------------------------------------------------------------

    assert config.target_spacing == spacing

    print("Target spacing: PASS")

    # --------------------------------------------------------------
    # Verify target shape
    # --------------------------------------------------------------

    assert config.target_shape == (
        64,
        256,
        256
    )

    print("target shape: PASS")

    # --------------------------------------------------------------
    # Verify default normalization
    # --------------------------------------------------------------

    assert config.normal_intensity is True

    print("Intensity normalization default: PASS")

    # --------------------------------------------------------------
    # Verify target spacing
    # --------------------------------------------------------------

    assert config.padding_value == 0.0

    print("Padding value default: PASS")

    # --------------------------------------------------------------
    # Verify invalid target shape
    # --------------------------------------------------------------

    try:
        ProcessingConfig(
            target_spacing=spacing,
            target_shape=(64, 256)
        )

    except ValueError:
        print("Invalid target shape: PASS")

    else:
        raise AssertionError("Inavalid target shape was accepted")


    # --------------------------------------------------------------
    # Verify zero dimension
    # --------------------------------------------------------------

    try:
        ProcessingConfig(
            target_spacing=spacing,
            target_shape=(64, 0, 256)
        )

    except ValueError:
        print("Invalid target dimension: PASS")

    else:
        raise AssertionError("Inavalid target dimension was accepted")

    # --------------------------------------------------------------
    # Verify negative dimension
    # --------------------------------------------------------------


    try:
        ProcessingConfig(
            target_spacing=spacing,
            target_shape=(64, -256, 256)
        )

    except ValueError:
        print("Negative target dimension: PASS")

    else:
        raise AssertionError("Negative target dimension was accepted")

    # --------------------------------------------------------------
    # Verify immutability
    # --------------------------------------------------------------   

    try:
        config.target_spacing = (
            32,
            128,
            128
        ) 

    except Exception as error:
        print(
            "Immutability test raised",
            type(error).__name__
        )

    else:
        raise AssertionError("PreprocessingConfig is mutable")

    print("Immutability: PASS")


    print()
    print("=" * 70)
    print("PRESPROCESSING CONFIG TEST PASSED")
    print("=" * 70)    

if __name__ == "__main__":
    main()
        