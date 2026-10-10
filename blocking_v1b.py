import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

BASE = Path("dataset/train")


# =========================================================
# 1. FAST NORMALIZATION
# =========================================================

def normalize_series(series):
    return (
        series.fillna("")
        .astype("string")
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# =========================================================
# 2. LOAD DATA
# =========================================================

print("Loading datasets...")

s1 = pd.read_csv(
    BASE / "train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s2 = pd.read_csv(
    BASE / "train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s3 = pd.read_csv(
    BASE / "train_source3.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

gt = pd.read_csv(
    BASE / "train_ground_truth.tsv",
    sep="\t"
)

print("Loaded.")


# =========================================================
# 3. FAST NORMALIZATION
# =========================================================

print("Normalizing S1...")

s1["norm_name"] = normalize_series(s1["business_name"])
s1["norm_address"] = normalize_series(s1["business_address"])

print("Normalizing S2...")

s2["norm_name"] = normalize_series(s2["business_name"])
s2["norm_address"] = normalize_series(s2["business_address"])

print("Normalizing S3...")

s3["norm_name"] = normalize_series(s3["business_name"])
s3["norm_address"] = normalize_series(s3["business_address"])

print("Normalization complete.")


# =========================================================
# 4. BUILD INDEX
# =========================================================

def build_index(df, source_name):

    print(f"Building {source_name} index...")

    name_index = defaultdict(list)
    address_index = defaultdict(list)

    for row in df.itertuples(index=False):

        country = row.country
        entity_id = row.entity_id

        name = row.norm_name
        address = row.norm_address

        if name:
            name_index[(country, name)].append(entity_id)

        if address:
            address_index[(country, address)].append(entity_id)

    print(
        f"{source_name} index ready:",
        len(name_index),
        "name keys,",
        len(address_index),
        "address keys"
    )

    return name_index, address_index


s2_name, s2_address = build_index(s2, "S2")
s3_name, s3_address = build_index(s3, "S3")


# =========================================================
# 5. GROUND TRUTH
# =========================================================

print("Preparing ground truth...")

gt_dict = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"].fillna("")
    )
)

print("Ground truth ready.")


# =========================================================
# 6. CANDIDATE GENERATION
# =========================================================

print("\nGenerating candidates...")

total_true_matches = 0
recovered_true_matches = 0

total_candidates = 0
max_candidates = 0

num_s1 = len(s1)

for i, row in enumerate(s1.itertuples(index=False), start=1):

    candidates = set()

    key_name = (row.country, row.norm_name)
    key_address = (row.country, row.norm_address)

    # -----------------------------------------------------
    # Exact normalized name
    # -----------------------------------------------------

    if row.norm_name:

        candidates.update(
            s2_name.get(key_name, [])
        )

        candidates.update(
            s3_name.get(key_name, [])
        )

    # -----------------------------------------------------
    # Exact normalized address
    # -----------------------------------------------------

    if row.norm_address:

        candidates.update(
            s2_address.get(key_address, [])
        )

        candidates.update(
            s3_address.get(key_address, [])
        )

    candidate_count = len(candidates)

    total_candidates += candidate_count

    if candidate_count > max_candidates:
        max_candidates = candidate_count

    # -----------------------------------------------------
    # Compare with ground truth
    # -----------------------------------------------------

    true_matches = gt_dict.get(row.entity_id, "")

    if true_matches:

        true_ids = set(true_matches.split(","))

        total_true_matches += len(true_ids)

        recovered_true_matches += len(
            true_ids.intersection(candidates)
        )

    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    if i % 100000 == 0:

        current_recall = (
            recovered_true_matches / total_true_matches
            if total_true_matches
            else 0
        )

        print(
            f"Processed {i:,} / {num_s1:,} | "
            f"Candidates so far: {total_candidates:,} | "
            f"Recall so far: {current_recall * 100:.2f}%"
        )


# =========================================================
# 7. FINAL RESULTS
# =========================================================

print("\n" + "=" * 70)
print("BLOCKING V1B RESULTS")
print("=" * 70)

print(
    "Total true matches:",
    total_true_matches
)

print(
    "Recovered true matches:",
    recovered_true_matches
)

recall = (
    recovered_true_matches / total_true_matches
    if total_true_matches
    else 0
)

average_candidates = (
    total_candidates / num_s1
)

print(
    "Candidate recall:",
    round(recall * 100, 4),
    "%"
)

print(
    "Total generated candidates:",
    total_candidates
)

print(
    "Average candidates per S1:",
    round(average_candidates, 2)
)

print(
    "Maximum candidates for one S1:",
    max_candidates
)

print(
    "Total S1 entities:",
    num_s1
)

print("\nDONE")