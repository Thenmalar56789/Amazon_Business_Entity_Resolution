import pandas as pd
from pathlib import Path

BASE = Path("dataset/train")

gt = pd.read_csv(
    BASE / "train_ground_truth.tsv",
    sep="\t"
)

gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

all_matches = []

for value in gt["matched_entity_ids"]:
    if value:
        all_matches.extend(value.split(","))

matches = pd.Series(all_matches, name="entity_id")

print("=" * 70)
print("TOTAL MATCH LINKS:", len(matches))
print("=" * 70)

counts = matches.value_counts()

print("\nUnique matched S2/S3 IDs:", counts.index.nunique())
print("Matched IDs appearing once:", (counts == 1).sum())
print("Matched IDs appearing more than once:", (counts > 1).sum())
print("Maximum times one ID appears:", counts.max())

print("\nRepeated IDs:")
print(counts[counts > 1].head(30))

print("\nPrefix distribution:")
print(matches.str[:3].value_counts())

print("\nDONE")