from pathlib import Path

import torch

from ish_knee.data.paths import DatasetPaths
from ish_knee.data.metadata import MetadataReader

from ish_knee.preprocessing.series_selector import SeriesSelector
from ish_knee.preprocessing.volume.preprocessing_config import PreprocessingConfig
from ish_knee.preprocessing.volume.voxel_spacing import VoxelSpacing
from ish_knee.preprocessing.volume.volume_preprocessor import VolumePreprocessor

from ish_knee.dataset.dataset_sample_builder import DatasetSampleBuilder
from ish_knee.dataset.labeled_dataset_builder import LabeledDatasetBuilder
from ish_knee.dataset.knee_mri_processor import KneeMRIProcessor

from ish_knee.model.model_sample_builder import ModelSampleBuilder
from ish_knee.model.model_dataset import ModelDataset
from ish_knee.model.pytorch_knee_dataset import PyTorchKneeDataset

def main():

        print("=" * 70)
        print("PYTORCH KNEE DATASET TEST")
        print("=" * 70)

        # ----------------------------------------------------------
        # Dataset metadata
        # ----------------------------------------------------------
    
        dataset_root = Path(
            r"D:\RSNA knee abnormality detection 2026"
        )
    
        paths = DatasetPaths.from_root(dataset_root)
    
        metadata = MetadataReader(paths)
        metadata.validate()

        series_selector = SeriesSelector(paths.train_series_dir)

        dataset_sample_builder = DatasetSampleBuilder(metadata, series_selector)

        labeled_builder = LabeledDatasetBuilder(dataset_sample_builder)

        samples = labeled_builder.build_all()

        print("Labeled studies:", len(samples))
        assert len(samples) == 58

        # ----------------------------------------------------------
        # Volume preprocessing
        # ---------------------------------------------------------- 

        config = PreprocessingConfig(target_spacing=VoxelSpacing(1.0,1.0,1.0), target_shape=(64,64,64))

        preprocessor = VolumePreprocessor(config)
        mri_processor = KneeMRIProcessor(preprocessor)

        # ----------------------------------------------------------
        # Model dataset
        # ---------------------------------------------------------- 

        model_sample_builder = ModelSampleBuilder(mri_processor)

        model_dataset = ModelDataset(samples, model_sample_builder)

        # ----------------------------------------------------------
        # Pytorch adapter
        # ----------------------------------------------------------  

        torch_dataset = PyTorchKneeDataset(model_dataset)

        print("pytorch dataset length:", len(torch_dataset))
        assert len(torch_dataset) == 58

        # ----------------------------------------------------------
        # Retrieve and process one real study
        # ----------------------------------------------------------  
        
        sagittal, coronal, axial, target = torch_dataset[0]

        # ----------------------------------------------------------
        # Validate tensor types
        # ----------------------------------------------------------  

        assert isinstance(sagittal, torch.Tensor)
        assert isinstance(coronal, torch.Tensor)
        assert isinstance(axial, torch.Tensor)
        assert isinstance(target, torch.Tensor)

        # ----------------------------------------------------------
        # Validate shapes
        # ----------------------------------------------------------  

        assert tuple(sagittal.shape) == (64, 64, 64)
        assert tuple(coronal.shape) == (64, 64, 64)
        assert tuple(axial.shape) == (64, 64, 64)
        assert tuple(target.shape) == (12,)

        # ----------------------------------------------------------
        # Validate data types
        # ----------------------------------------------------------          

        assert sagittal.dtype == torch.float32
        assert coronal.dtype == torch.float32
        assert axial.dtype == torch.float32
        assert target.dtype == torch.float32

        print()
        print("Sagittal tensor:", tuple(sagittal.shape))
        print("Coronal tensor:", tuple(coronal.shape))
        print("Axial tensor:", tuple(axial.shape))
        print("Target tensor:", tuple(target.shape))
        print("Tensor dtype:", sagittal.dtype)
        print("Target values:", target.tolist())

        print()
        print("PYTORCH KNEE DATASET TEST PASSED")
        print("=" * 70)

if __name__ == "__main__":
        main()