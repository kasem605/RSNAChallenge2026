from pathlib import Path
from typing import Optional

import numpy as np
import pydicom

from.models import SeriesMetadata

def _get_value(ds, name: str, default=None):

    value = getattr(ds, name, default)

    if value is None:
        return default

    return value

def _to_float(value) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _to_int(value) -> Optional[int]:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _to_tuple(value):

    if value is None:
        return None

    try:
        return tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None

def infer_plane(image_orientation):

    if image_orientation is None:
        return None

    if(len(image_orientation) != 6):
        return None

    row = np.asarray(image_orientation[:3],dtype=np.float64)
    column = np.asarray(image_orientation[3:], dtype=np.float64)

    normal = np.cross(row, column)

    axis = int(np.argmax(np.abs(normal)))

    if axis == 0:
        return "sagittal"

    if axis == 1:
        return "coronal"

    if axis == 2:
        return "axial"

    return None

def read_dicom_header(path: Path):

    return pydicom.dcmread(
        str(path),
        stop_before_pixels=True,
        force=True,
    )

def read_series_metadata(
        study_uid: str,
        series_uid: str,
        series_path: Path,
) -> SeriesMetadata:

    files = sorted(series_path.glob("*.dcm"))

    if not files:
        raise FileNotFoundError(
            f"No DICOM files found in {series_path}"
        )

    first_file=files[0]

    ds = read_dicom_header(first_file)

    image_orientation = _to_tuple(
        _get_value(ds, "ImageOrientationPatient")
    )

    return SeriesMetadata(

        study_instance_uid=str(
            _get_value(
                ds,
                "StudyInstanceUID",
                study_uid
            )
        ),

        study_instance_uid=str(
            _get_value(
                ds,
                "SeriesInstanceUID",
                study_uid
            )
        ), 

        series_number=_to_int(
            _get_value(ds, "SeriesNumber")
        ),

        series_description=_get_value(
            ds, 
            "SeriesDescription"
        ),

        protocol_name=_get_value(
            ds, 
            "ProtocolName"
        ),

        sequence_name=_get_value(
            ds,
            "SequenceName"
        ),

        modality=_get_value(
            ds,
            "Modality"
        ),

        body_part_examined=_get_value(
            ds,
            "BodyPartExamined"
        ),

        manufacturer=_get_value(
            ds,
            "Manufacturer"
        ),

        manufacturer_model_name=_get_value(
            ds,
            "ManufacturerModelName"
        ),

        magnetic_field_strength=_get_value(
            ds,
            "MagneticFieldStrength"
        ),

        rows=_to_int(
            _get_value(ds, "Rows")
        ),

        columns=_to_int(
            _get_value(ds, "Columns")
        ),

        pixel_spacing=_to_tuple(
            _get_value(ds, "PixelSpacing")
        ),

        slice_thickness=_to_float(
            _get_value(ds, "SliceThickness")
        ),

        spacing_between_slices=_to_float(
            _get_value(ds, "SPacingBetweenSlices")
        ),

        image_orientation_patient=image_orientation,

        image_position_patient=_to_tuple(
            _get_value(ds, "ImagePositionPatient")
        ),

        photometric_interpretation=_get_value(
            ds,
            "PhotometricInterpretation"
        ),

        rescale_slope=_to_float(
            _get_value(ds, "RescaleSlope")
        ),

        rescale_intercept=_to_float(
            _get_value(ds, "RescaleIntercept")
        ),

        number_of_frames=_to_int(
            _get_value(ds, "NumberOfFrames")
        ),

        file_count=len(files),

        inferred_plane=infer_plane(
            image_orientation
        ),
    )





