
from pathlib import Path

import pandas as pd

from ..dataset.dataset_sample import DatasetSample
from ..dataset.knee_labels import KneeLabels
from ..dataset.knee_study_sample import KneeStudySample
from .metadata import MetadataReader
from ..preprocessing.series_selector import SeriesSelector


LABEL_COLUMNS = {
    "acl": "ACL",
    "mcl": "MCL",
    "medial_meniscus": "Medial Meniscus",
    "lateral_meniscus": "Lateral Meniscus",
    "medial_oa": "Medial OA",
    "lateral_oa": "Lateral OA",
    "pf_oa": "PF OA",
    "effusion": "Effusion",
    "synovitis": "Synovitis",
    "bakers": "Baker's",
    "contusion": "Contusion",
    "fracture": "Fracture",
}


class LabeledSampleBuilder:
    """Build DatasetSample objects from fully labeled studies."""

    def __init__(
        self,
        metadata: MetadataReader,
        series_selector: SeriesSelector,
    ) -> None:
        self._metadata = metadata
        self._series_selector = series_selector

    def build(self) -> list[DatasetSample]:
        train = self._metadata.train

        required_columns = {
            "StudyInstanceUID",
            *LABEL_COLUMNS.values(),
        }
        missing = required_columns - set(train.columns)

        if missing:
            raise ValueError(
                f"train.csv is missing columns: {sorted(missing)}"
            )

        # Only studies with all 12 known labels are eligible.
        labeled = train.dropna(
            subset=list(LABEL_COLUMNS.values())
        ).copy()

        samples: list[DatasetSample] = []
        seen_uids: set[str] = set()

        for _, row in labeled.iterrows():
            study_uid = str(row["StudyInstanceUID"])

            if not study_uid or study_uid == "nan":
                raise ValueError("A labeled study has an empty UID.")

            if study_uid in seen_uids:
                raise ValueError(
                    f"Duplicate labeled study UID: {study_uid}"
                )

            seen_uids.add(study_uid)

            values = {}
            for attribute, column in LABEL_COLUMNS.items():
                value = row[column]

                # Reject unexpected values rather than silently
                # converting them to positive or negative labels.
                if value not in (0, 1, 0.0, 1.0):
                    raise ValueError(
                        f"Invalid label {value!r} in column "
                        f"{column!r}, study {study_uid}"
                    )

                values[attribute] = int(value)

            labels = KneeLabels(**values)

            study_series = self._metadata.get_study_series(study_uid)
            selection = self._series_selector.select(
                study_uid,
                study_series,
            )

            study = KneeStudySample(
                study_instance_uid=study_uid,
                sagittal=selection.sagittal,
                coronal=selection.coronal,
                axial=selection.axial,
            )

            samples.append(
                DatasetSample(
                    study=study,
                    labels=labels,
                )
            )

        return samples
