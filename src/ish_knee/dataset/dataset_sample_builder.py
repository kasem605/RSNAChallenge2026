from ish_knee.data.metadata import MetadataReader
from ish_knee.preprocessing.series_selector import SeriesSelector

from .dataset_sample import DatasetSample
from .knee_labels import KneeLabels
from .knee_study_sample import KneeStudySample

import pandas as pd

class DatasetSampleBuilder:

    """
    Builds a complete DatasetSample from the RSNA metadata.

    The builder connects:

        train.csv
            +
        train_series.csv
            +
        SeriesSelector

    into one study-level training sample.
    
    """
    
    LABELED_COLUMNS = [
            "ACL",
            "MCL",
            "Medial Meniscus",
            "Lateral Meniscus",
            "Medial OA",
            "Lateral OA",
            "PF OA",
            "Effusion",
            "Synovitis",
            "Baker's",
            "Contusion",
            "Fracture"
        ]
    def __init__(self, metadata: MetadataReader, series_selector: SeriesSelector) -> None:

        if not isinstance(metadata, MetadataReader):
            raise TypeError("metadata must be a MetadataReader instance")

        if not isinstance(series_selector, SeriesSelector):
            raise TypeError("series_selector must be a SeriesSelector instance") 

        self._metadata = metadata
        self._series_selector = series_selector

    def build(self, study_instance_uid: str) -> DatasetSample:

        if not study_instance_uid:
            raise ValueError("study_instance_uid cannot be empty.")

        study_rows = self._metadata.train[
            self._metadata.train["StudyInstanceUID"] == study_instance_uid
        ]    

        if study_rows.empty:
            raise ValueError(f"StudyInstanceUID not found: {study_instance_uid}")
        
        study_row = study_rows.iloc[0]
        
        missing_labels = [
            column
            for column in self.LABELED_COLUMNS
            if pd.isna(study_row[column])
        ]

        if missing_labels:
            raise ValueError(f"StudyInstanceUID has incomplete abnormality labels: {study_instance_uid}")
        
        series_rows = self._metadata.get_study_series(study_instance_uid)

        selection = self._series_selector.select(study_uid=study_instance_uid, series=series_rows)

        study = KneeStudySample(
            study_instance_uid=study_instance_uid,
            sagittal=selection.sagittal,
            coronal=selection.coronal,
            axial=selection.axial
        )

        labels = KneeLabels(
            acl=int(study_row["ACL"]),
            mcl=int(study_row["MCL"]),
            medial_meniscus=int(study_row["Medial Meniscus"]),
            lateral_meniscus=int(study_row["Lateral Meniscus"]),
            medial_oa=int(study_row["Medial OA"]),
            lateral_oa=int(study_row["Lateral OA"]),
            pf_oa=int(study_row["PF OA"]),
            effusion=int(study_row["Effusion"]),
            synovitis=int(study_row["Synovitis"]),
            bakers=int(study_row["Baker's"]),
            contusion=int(study_row["Contusion"]),
            fracture=int(study_row["Fracture"])          
        )

        return DatasetSample(study=study, labels=labels)
