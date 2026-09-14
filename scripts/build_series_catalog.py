from pathlib import Path
import sys

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from ish_knee.data import(
    DatasetPaths,
    MetadataReader,
    SeriesCatalogBuilder
)

def main():

    dataset_root =r"D:\RSNA knee abnormality detection 2026"

    paths = DatasetPaths.from_root(dataset_root)

    metadata=MetadataReader(paths)

    metadata.validate()

    builder = SeriesCatalogBuilder(paths, metadata)

    catalog=builder.build()

    output_path = (
        PROJECT_ROOT
        / "data"
        / "metadata"
        / "series_catalog.parquet"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalog.to_parquet(
        output_path,
        index=False,
    )

    print()
    print(f"Saved catalog to: {output_path}")

    print(output_path.exists())


    print()
    print(
        catalog[
            [       
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "Anatomical_PLane",
                "Fluid_Sensitive",
                "Fat_Suppression",
                "FileCount",
            ]
        ].head(20)
    )

    print()
    print("Catalog shape:")
    print(catalog.shape)

    print()
    print("Missing Series:")
    print((~catalog["SeriesExists"]).sum())

    print()
    print("Total Files")
    print(catalog["FileCount"].sum())

if __name__ == "__main__":
    main()