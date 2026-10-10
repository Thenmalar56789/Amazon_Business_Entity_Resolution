import pandas as pd
from pathlib import Path

BASE = Path("dataset")

files = {
    "train_source1": BASE / "train" / "train_source1.tsv",
    "train_source2": BASE / "train" / "train_source2.tsv",
    "train_source3": BASE / "train" / "train_source3.tsv",
    "ground_truth": BASE / "train" / "train_ground_truth.tsv",
    "test_source1": BASE / "test" / "test_source1.tsv",
    "test_source2": BASE / "test" / "test_source2.tsv",
    "test_source3": BASE / "test" / "test_source3.tsv",
}

print("=" * 70)
print("AMAZON ML CHALLENGE 2026 - DATASET RECONNAISSANCE")
print("=" * 70)

for name, path in files.items():

    print(f"\n{'=' * 25} {name.upper()} {'=' * 25}")

    if not path.exists():
        print("FILE NOT FOUND:", path)
        continue

    print("File size:", round(path.stat().st_size / (1024 ** 2), 2), "MB")

    df = pd.read_csv(path, sep="\t")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Columns:", list(df.columns))

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    if "country" in df.columns:
        print("\nCountry distribution:")
        print(df["country"].value_counts(dropna=False).head(20))

    if "business_name" in df.columns:
        print("\nBusiness name statistics:")
        print("Unique:", df["business_name"].nunique())
        print("Average length:", round(df["business_name"].fillna("").str.len().mean(), 2))

    if "business_address" in df.columns:
        print("\nBusiness address statistics:")
        print("Unique:", df["business_address"].nunique())
        print("Average length:", round(df["business_address"].fillna("").str.len().mean(), 2))

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)