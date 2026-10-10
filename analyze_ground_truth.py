import pandas as pd
from pathlib import Path

path = Path("dataset/train/train_ground_truth.tsv")

gt = pd.read_csv(path, sep="\t")

# Empty match lists are represented as NaN
gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

def count_source_matches(x, prefix):
    if not x:
        return 0
    return sum(1 for item in x.split(",") if item.startswith(prefix))

gt["s2_count"] = gt["matched_entity_ids"].apply(
    lambda x: count_source_matches(x, "S2-")
)

gt["s3_count"] = gt["matched_entity_ids"].apply(
    lambda x: count_source_matches(x, "S3-")
)

gt["total_matches"] = gt["s2_count"] + gt["s3_count"]

print("=" * 70)
print("GROUND TRUTH ANALYSIS")
print("=" * 70)

print("\nTotal Source 1 entities:", len(gt))

print("\nTotal entities with ZERO matches:")
print((gt["total_matches"] == 0).sum())

print("\nTotal entities with at least one match:")
print((gt["total_matches"] > 0).sum())

print("\nTotal entities with S2 matches:")
print((gt["s2_count"] > 0).sum())

print("\nTotal entities with S3 matches:")
print((gt["s3_count"] > 0).sum())

print("\nEntities with BOTH S2 and S3 matches:")
print(((gt["s2_count"] > 0) & (gt["s3_count"] > 0)).sum())

print("\nDistribution of TOTAL matches per S1:")
print(gt["total_matches"].value_counts().sort_index().head(30))

print("\nDistribution of S2 matches:")
print(gt["s2_count"].value_counts().sort_index().head(20))

print("\nDistribution of S3 matches:")
print(gt["s3_count"].value_counts().sort_index().head(20))

print("\nMaximum S2 matches for one S1:")
print(gt["s2_count"].max())

print("\nMaximum S3 matches for one S1:")
print(gt["s3_count"].max())

print("\nMaximum TOTAL matches for one S1:")
print(gt["total_matches"].max())

print("\nSample S1 entities with ZERO matches:")
print(
    gt[gt["total_matches"] == 0]
    .head(10)
    .to_string(index=False)
)

print("\nSample S1 entities with MANY matches:")
print(
    gt.sort_values("total_matches", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)