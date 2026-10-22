import pandas as pd

df = pd.read_csv(
    "outputs/processed_labeled_dataset/processed_volumes.csv"
)

print(df.columns.tolist())
print(df.head(3).to_string(index=False))