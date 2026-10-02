from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from dataclasses import FrozenInstanceError

def main():

    print()
    print("=" * 70)
    print("VOXEL SPACING TEST")
    print("=" * 70)

    spacing = VoxelSpacing(
        spacing_axis_0=3.0,
        spacing_axis_1=0.5,
        spacing_axis_2=0.5
    )

    print()
    print("Spacing:")
    print(spacing)

    # ----------------------------------------------------------------
    # Verify individual values
    # ----------------------------------------------------------------

    assert spacing.spacing_axis_0 == 3.0
    assert spacing.spacing_axis_1 == 0.5
    assert spacing.spacing_axis_2 == 0.5

    print()
    print("Individual values: PASS")

    # ----------------------------------------------------------------
    # Verify tuple
    # ----------------------------------------------------------------   

    assert spacing.as_tuple ==(
        3.0,
        0.5,
        0.5
    )

    print("IndividuaL values: PASS")

    # ----------------------------------------------------------------
    # Verify immutability
    # ----------------------------------------------------------------

    try:
        spacing.spacing_axis_0 = 2.0

    except FrozenInstanceError:
        print("Immutability: PASS")

    else:
        raise AssertionError("Voxelspacing is mutable")

    print()
    print("=" * 70)
    print("VOXEL SPACING TEST PASSED")
    print("=" * 70)    

if __name__ == "__main__":
    main()