import pandas as pd
import re
from pathlib import Path
from collections import defaultdict, Counter

BASE = Path("dataset/train")

MIN_TOKEN_LENGTH = 4
MAX_TOKEN_FREQUENCY = 200
MIN_SHARED_TOKENS = 2


# =========================================================
# 1. NORMALIZATION
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


def get_tokens(text):
    if not text:
        return []

    return {
        token
        for token in text.split()
        if len(token) >= MIN_TOKEN_LENGTH
    }


# =========================================================
# 2. LOAD
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
# 3. NORMALIZE
# =========================================================

print("Normalizing...")

for df in [s1, s2, s3]:

    df["norm_name"] = normalize_series(
        df["business_name"]
    )

    df["norm_address"] = normalize_series(
        df["business_address"]
    )

print("Normalization complete.")


# =========================================================
# 4. EXACT BLOCK INDEXES
# =========================================================

def build_exact_indexes(df, source):

    print(f"Building exact indexes for {source}...")

    name_index = defaultdict(list)
    address_index = defaultdict(list)

    for row in df.itertuples(index=False):

        if row.norm_name:

            name_index[
                (row.country, row.norm_name)
            ].append(row.entity_id)

        if row.norm_address:

            address_index[
                (row.country, row.norm_address)
            ].append(row.entity_id)

    return name_index, address_index


s2_name, s2_address = build_exact_indexes(s2, "S2")
s3_name, s3_address = build_exact_indexes(s3, "S3")


# =========================================================
# 5. TOKEN FREQUENCY
# =========================================================

def count_tokens(df, source):

    print(f"Counting tokens for {source}...")

    counts = Counter()

    for i, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):

        tokens = get_tokens(
            f"{row.norm_name} {row.norm_address}"
        )

        for token in tokens:

            counts[
                (row.country, token)
            ] += 1

        if i % 1000000 == 0:

            print(
                f"{source}: "
                f"{i:,} / {len(df):,}"
            )

    return counts


s2_counts = count_tokens(s2, "S2")
s3_counts = count_tokens(s3, "S3")


# =========================================================
# 6. TOKEN INDEX
# =========================================================

def build_token_index(
    df,
    counts,
    source
):

    print(f"Building token index for {source}...")

    index = defaultdict(list)

    for i, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):

        tokens = get_tokens(
            f"{row.norm_name} {row.norm_address}"
        )

        for token in tokens:

            key = (row.country, token)

            if counts[key] <= MAX_TOKEN_FREQUENCY:

                index[key].append(
                    row.entity_id
                )

        if i % 1000000 == 0:

            print(
                f"{source}: "
                f"{i:,} / {len(df):,}"
            )

    return index


s2_tokens = build_token_index(
    s2,
    s2_counts,
    "S2"
)

s3_tokens = build_token_index(
    s3,
    s3_counts,
    "S3"
)


# =========================================================
# 7. GROUND TRUTH
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
# 8. GENERATE SELECTIVE CANDIDATES
# =========================================================

print("\nGenerating selective candidates...")

total_true = 0
recovered = 0

total_candidates = 0
max_candidates = 0

num_s1 = len(s1)


for i, row in enumerate(
    s1.itertuples(index=False),
    start=1
):

    candidates = set()

    # -----------------------------------------------------
    # A. Exact name
    # -----------------------------------------------------

    if row.norm_name:

        key = (
            row.country,
            row.norm_name
        )

        candidates.update(
            s2_name.get(key, [])
        )

        candidates.update(
            s3_name.get(key, [])
        )


    # -----------------------------------------------------
    # B. Exact address
    # -----------------------------------------------------

    if row.norm_address:

        key = (
            row.country,
            row.norm_address
        )

        candidates.update(
            s2_address.get(key, [])
        )

        candidates.update(
            s3_address.get(key, [])
        )


    # -----------------------------------------------------
    # C. Token overlap >= 2
    # -----------------------------------------------------

    tokens = get_tokens(
        f"{row.norm_name} {row.norm_address}"
    )

    token_hits = Counter()

    for token in tokens:

        key = (
            row.country,
            token
        )

        for entity_id in s2_tokens.get(key, []):

            token_hits[entity_id] += 1

        for entity_id in s3_tokens.get(key, []):

            token_hits[entity_id] += 1


    for entity_id, count in token_hits.items():

        if count >= MIN_SHARED_TOKENS:

            candidates.add(entity_id)


    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    count = len(candidates)

    total_candidates += count

    max_candidates = max(
        max_candidates,
        count
    )


    # -----------------------------------------------------
    # Recall
    # -----------------------------------------------------

    true_matches = gt_dict.get(
        row.entity_id,
        ""
    )

    if true_matches:

        true_ids = set(
            true_matches.split(",")
        )

        total_true += len(true_ids)

        recovered += len(
            true_ids.intersection(candidates)
        )


    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    if i % 100000 == 0:

        recall = (
            recovered / total_true
            if total_true
            else 0
        )

        print(
            f"Processed {i:,} / {num_s1:,} | "
            f"Candidates: {total_candidates:,} | "
            f"Recall: {recall * 100:.2f}%"
        )


# =========================================================
# 9. RESULTS
# =========================================================

print("\n" + "=" * 70)
print("BLOCKING V3 RESULTS")
print("=" * 70)

print(
    "Total true matches:",
    total_true
)

print(
    "Recovered true matches:",
    recovered
)

recall = (
    recovered / total_true
    if total_true
    else 0
)

print(
    "Candidate recall:",
    round(
        recall * 100,
        4
    ),
    "%"
)

print(
    "Total generated candidates:",
    total_candidates
)

print(
    "Average candidates per S1:",
    round(
        total_candidates / num_s1,
        2
    )
)

print(
    "Maximum candidates for one S1:",
    max_candidates
)

print("\nDONE")