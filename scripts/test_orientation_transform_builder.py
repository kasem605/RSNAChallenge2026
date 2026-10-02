from ish_knee.preprocessing.volume.orientation_info import OrientationInfo
from ish_knee.preprocessing.orientation_transform_builder import OrientationTransformBuilder

def main():

    print("=" * 70)
    print("ORIENTATION TRANSFORM BUILDER TEST")
    print("=" * 70)

    builder = OrientationTransformBuilder()

    # ----------------------------------------------------------------
    # Test1: Axial
    # ----------------------------------------------------------------

    orientation = OrientationInfo(
        row_direction=(1.0, 0.0, 0.0),
        column_direction=(0.0, 1.0, 0.0),
        slice_normal=(0.0, 0.0, 1.0),
        anatomical_plane="axial"
    )

    transformation = builder.build(orientation)

    print("Axial axis order: ", transformation.axis_order)

    assert  transformation.axis_order == (2, 0, 1)

    print("Axial Test:  PASSED")

    # ----------------------------------------------------------------
    # Test2: Coronal
    # ----------------------------------------------------------------

    orientation = OrientationInfo(
        row_direction=(1.0, 0.0, 0.0),
        column_direction=(0.0, 0.0, 1.0),
        slice_normal=(0.0, -1.0, 0.0),
        anatomical_plane="axial"
    )

    transformation = builder.build(orientation)

    print("Coronal axis order: ", transformation.axis_order)

    assert  transformation.axis_order == (1, 0, 2)

    print("Coronal Test:  PASSED")

    # ----------------------------------------------------------------
    # Test3: Sagittal
    # ----------------------------------------------------------------

    orientation = OrientationInfo(
        row_direction=(.0, 1.0, 0.0),
        column_direction=(0.0, 0.0, 1.0),
        slice_normal=(1.0, 0.0, 0.0),
        anatomical_plane="sagittal"
    )

    transformation = builder.build(orientation)

    print("Sagittal axis order: ", transformation.axis_order)

    assert  transformation.axis_order == (0, 1, 2)

    print("Sagittal Test:  PASSED")    

    # ----------------------------------------------------------------
    # Verify flips are currently disabled
    # ----------------------------------------------------------------

    assert transformation.flip_axis_0 is False
    assert transformation.flip_axis_1 is False
    assert transformation.flip_axis_2 is False

    print("Initial flip settings: PASSED")

    print()
    print("ORIENTATION TRANSFORM BUILDER TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()