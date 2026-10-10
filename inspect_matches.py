import pandas as pd
from pathlib import Path

BASE = Path("dataset/train")

s1 = pd.read_csv(BASE / "train_source1.tsv", sep="\t")
s2 = pd.read_csv(BASE / "train_source2.tsv", sep="\t")
s3 = pd.read_csv(BASE / "train_source3.tsv", sep="\t")
gt = pd.read_csv(BASE / "train_ground_truth.tsv", sep="\t")

# Select examples with different numbers of matches
gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
gt["match_count"] = gt["matched_entity_ids"].apply(
    lambda x: 0 if not x else len(x.split(","))
)

examples = []

for count in [0, 1, 3, 5, 11]:
    row = gt[gt["match_count"] == count].iloc[0]
    examples.append(row["source1_entity_id"])

for s1_id in examples:

    print("\n" + "=" * 100)
    print("SOURCE 1:", s1_id)
    print("=" * 100)

    r1 = s1[s1["entity_id"] == s1_id]
    print("\nS1 RECORD:")
    print(r1.to_string(index=False))

    matched = gt.loc[
        gt["source1_entity_id"] == s1_id,
        "matched_entity_ids"
    ].iloc[0]

    if not matched:
        print("\nNO MATCHES")
        continue

    ids = matched.split(",")

    s2_ids = [x for x in ids if x.startswith("S2-")]
    s3_ids = [x for x in ids if x.startswith("S3-")]

    if s2_ids:
        print("\nTRUE S2 MATCHES:")
        print(
            s2[s2["entity_id"].isin(s2_ids)]
            .to_string(index=False)
        )

    if s3_ids:
        print("\nTRUE S3 MATCHES:")
        print(
            s3[s3["entity_id"].isin(s3_ids)]
            .to_string(index=False)
        )

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)