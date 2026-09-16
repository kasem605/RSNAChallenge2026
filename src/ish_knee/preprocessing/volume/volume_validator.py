from ..Dicom.dicom_volume import DicomVolume

import numpy as np

class VolumeValidator:

    """
    Validates the structure and numeric contents
    of a DICOM MRI volume
    """

    def validate(self, volume: DicomVolume) -> None:
        """
        validates a DICOM volume.

        Raises:
            TypeError:
                If the volume or underlying data
                has an invalid type.

            ValueError:
                If the volume has an invalid shape
                or contains invalid numeric values.
        """

        # -------------------------------------------------------------------
        # Validate DicomVolume
        # -------------------------------------------------------------------

        if not isinstance(volume, DicomVolume):
            raise TypeError("Expected a DicomVolume instance.")

        # -------------------------------------------------------------------
        # Validate Numpy array
        # -------------------------------------------------------------------     

        if not isinstance(
            volume.volume,
            np.ndarray
        ):
               
            if not isinstance(volume, DicomVolume):
                raise TypeError("DicomVolume.volume must be a Numpy array.")

        # -------------------------------------------------------------------
        # Validate dimensionality
        # -------------------------------------------------------------------     

        if volume.volume.ndim != 3:
                        
            if not isinstance(volume, DicomVolume):
                raise TypeError("MRI volume must be 3-dimensional.")

        # -------------------------------------------------------------------
        # Validate dimensions
        # -------------------------------------------------------------------     

        depth, rows, columns = volume.volume.shape

        if depth <= 0:
            raise ValueError("MRI volume contains no slice")

        if rows <= 0 or columns <= 0:
            raise ValueError("MRI volume contains invalid image dimensions.")

        # -------------------------------------------------------------------
        # Validate numeric values
        # -------------------------------------------------------------------  

        if not np.isfinite(volume.volume).all():

            raise ValueError("MRI volume contains Nan or infinite values.")   

       