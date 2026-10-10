import pandas as pd
from pathlib import Path

BASE = Path("dataset/train")

s1 = pd.read_csv(BASE / "train_source1.tsv", sep="\t")
s2 = pd.read_csv(BASE / "train_source2.tsv", sep="\t")
s3 = pd.read_csv(BASE / "train_source3.tsv", sep="\t")
gt = pd.read_csv(BASE / "train_ground_truth.tsv", sep="\t")

# Create fast ID -> country dictionaries
s1_country = dict(zip(s1["entity_id"], s1["country"]))
s2_country = dict(zip(s2["entity_id"], s2["country"]))
s3_country = dict(zip(s3["entity_id"], s3["country"]))

same_country = 0
different_country = 0
total = 0

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]
    matched = row["matched_entity_ids"]

    if pd.isna(matched) or not matched:
        continue

    source1_country = s1_country[s1_id]

    for match_id in matched.split(","):

        if match_id.startswith("S2-"):
            match_country = s2_country[match_id]
        else:
            match_country = s3_country[match_id]

        total += 1

        if source1_country == match_country:
            same_country += 1
        else:
            different_country += 1

print("=" * 70)
print("COUNTRY CONSISTENCY")
print("=" * 70)

print("Total true matches:", total)
print("Same country:", same_country)
print("Different country:", different_country)

print(
    "Same-country percentage:",
    round(100 * same_country / total, 4),
    "%"
)

print(
    "Different-country percentage:",
    round(100 * different_country / total, 4),
    "%"
)

print("\nDONE")