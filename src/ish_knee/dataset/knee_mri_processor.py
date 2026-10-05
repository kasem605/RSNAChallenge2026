from pathlib import Path

from .dataset_sample import DatasetSample
from .knee_mri_sample import KneeMRISample

from ..preprocessing.dicom.dicom_reader import DicomSeriesReader
from ..preprocessing.dicom.dicom_volume_builder import DicomVolumeBuilder
from ..preprocessing.volume.spacing_analyzer import SpacingAnalyzer
from ..preprocessing.volume.volume_preprocessor import VolumePreprocessor
from ..preprocessing.series_selector import SelectedSeries

class KneeMRIProcessor:
    """
    Processes the three MRI planes for one knee study

    Processing pipeline:
            |
    SelectedSeries
            |
    DicomSeriesReader
            |
    DicomVolumeBuilder
            |
    SpacingAnalyzer
            |
    VolumePreprocessor
            |
    Numpy volume

    The final result is returned as a KneeMRISample
    """

    def __init__(self, preprocessor: VolumePreprocessor)-> None:

        if not isinstance(preprocessor, VolumePreprocessor):
            raise TypeError("preprocessor must be a VolumePreprocessor instance")

        self._series_reader = DicomSeriesReader()
        self._volume_builder = DicomVolumeBuilder()        
        self._spacing_analyzer = SpacingAnalyzer()
        self._preprocessor = preprocessor

    def process(self, sample: DatasetSample)-> KneeMRIProcessor:

        if not isinstance(sample, DatasetSample):
            raise TypeError("sample must be a DatasetSample instance")

        if not sample.study.is_complete:
            raise ValueError(f"Study {sample.study_instance_uid} does not contain all three MRI planes")

        sagittal = self._process_series(sample.study.sagittal)
        coronal = self._process_series(sample.study.coronal)
        axial = self._process_series(sample.study.axial)

        return KneeMRISample(
            study_instance_uid=sample.study_instance_uid,
            sagittal=sagittal.volume,
            coronal=coronal.volume,
            axial=axial.volume
        )

    def _process_series(self, selected_series: SelectedSeries):

        series_path = Path(selected_series.series_path)

        series = self._series_reader.read(series_path)

        volume = self._volume_builder.build(series)

        spacing = self._spacing_analyzer.analyze(series)

        processed_volume = self._preprocessor.preprocess(volume, spacing)

        return processed_volume