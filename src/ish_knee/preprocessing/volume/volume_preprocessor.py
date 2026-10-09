from ..dicom.dicom_volume import DicomVolume
from .preprocessing_config import PreprocessingConfig
from .spacing_resampler import SpacingResampler
from .volume_resizer import VolumeResizer
from .volume_padder import VolumePadder
from .intensity_normalizer import IntensityNormalizer

class VolumePreprocessor:

    """
    Coordinates the preprocessing operations required
    to preapare a DICOM MRI volume for model input

    The individual preprocessing remain in
    their own classes. The class controls their order
    """

    def __init__(self, config: PreprocessingConfig) ->None:

        if not isinstance(config, PreprocessingConfig):
            raise TypeError("Config must be a PreprocessingConfig instance")

        self._config = config

        self._spacing_resampler = SpacingResampler()

        self._volume_resizer = VolumeResizer()

        self._volume_padder = VolumePadder()

        self._intensity_normalizer = IntensityNormalizer()


    def preprocess(self, volume: DicomVolume, current_spacing) -> DicomVolume:

        """
        Preprocess a DICOM volume.

        Process order:
            1. Validate input
            2. Resample voxel spacing
            3. Resize if necessary
            4. Pad to target shape

        Parameters
        ----------
        volume:
            Input DICOM volume

        Current_spacing:
            Current voxel spacing of the volume.

        Returns
        -------
        DicomVolume
            Preprocessed volume.

        """

        if not isinstance(volume, DicomVolume):
            raise TypeError("volume must bea DicomVolume instance")

        # --------------------------------------------------------
        # Step 1: Resample voxel spacing
        # --------------------------------------------------------

        processed_volume = (
            self._spacing_resampler.resample(
                volume=volume,
                current_spacing=current_spacing,
                target_spacing=self._config.target_spacing
            )
        )

        # --------------------------------------------------------
        # Step 2: Resize
        # --------------------------------------------------------       

        current_shape = processed_volume.volume.shape
        target_shape = self._config.target_shape

        resize_shape = tuple(
            min(current, target)
            for current, target in zip(
                current_shape,
                target_shape
            )
        )

        if resize_shape != current_shape:

            processed_volume = (
                self._volume_resizer.resize(
                    volume=processed_volume,
                    target_shape=resize_shape
                )
            )

        # ----------------------------------------------------------------
        # Step 3: Pad to target shape
        # ----------------------------------------------------------------

        if processed_volume.volume.shape != target_shape:
            processed_volume =(
                self._volume_padder.pad(
                    volume = processed_volume,
                    target_shape=target_shape,
                    constant_value=self._config.padding_value
                )
            )

        # ----------------------------------------------------------------
        # Step 4: Normalize intensity
        # ----------------------------------------------------------------

        if self._config.normalize_intensity:
            processed_volume = (
                self._intensity_normalizer.normalize(
                    volume=processed_volume
                )
            )

        return processed_volume