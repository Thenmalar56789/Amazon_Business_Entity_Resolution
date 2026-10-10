import pandas as pd
import re
from pathlib import Path
from collections import defaultdict, Counter

BASE = Path("dataset/train")

# Maximum number of records allowed for one token.
# Very common tokens are ignored.
MAX_TOKEN_FREQUENCY = 200

# Ignore very short tokens.
MIN_TOKEN_LENGTH = 4


# =========================================================
# 1. TEXT NORMALIZATION
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

    return [
        token
        for token in text.split()
        if len(token) >= MIN_TOKEN_LENGTH
    ]


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
# 3. NORMALIZE
# =========================================================

print("Normalizing...")

for df in [s1, s2, s3]:

    df["norm_name"] = normalize_series(df["business_name"])

    df["norm_address"] = normalize_series(
        df["business_address"]
    )

print("Normalization complete.")


# =========================================================
# 4. COUNT TOKEN FREQUENCIES
# =========================================================

def count_tokens(df, source_name):

    print(f"\nCounting tokens in {source_name}...")

    counter = Counter()

    for i, row in enumerate(df.itertuples(index=False), start=1):

        # Combine name + address
        text = f"{row.norm_name} {row.norm_address}"

        tokens = set(get_tokens(text))

        for token in tokens:
            counter[(row.country, token)] += 1

        if i % 500000 == 0:
            print(
                f"{source_name}: "
                f"{i:,} / {len(df):,}"
            )

    print(
        f"{source_name} token keys:",
        len(counter)
    )

    return counter


s2_token_counts = count_tokens(s2, "S2")
s3_token_counts = count_tokens(s3, "S3")


# =========================================================
# 5. BUILD TOKEN INDEX
# =========================================================

def build_token_index(df, token_counts, source_name):

    print(f"\nBuilding {source_name} token index...")

    token_index = defaultdict(list)

    for i, row in enumerate(df.itertuples(index=False), start=1):

        text = f"{row.norm_name} {row.norm_address}"

        tokens = set(get_tokens(text))

        for token in tokens:

            key = (row.country, token)

            # Ignore extremely common tokens
            if token_counts[key] <= MAX_TOKEN_FREQUENCY:

                token_index[key].append(
                    row.entity_id
                )

        if i % 500000 == 0:
            print(
                f"{source_name}: "
                f"{i:,} / {len(df):,}"
            )

    print(
        f"{source_name} index keys:",
        len(token_index)
    )

    return token_index


s2_token_index = build_token_index(
    s2,
    s2_token_counts,
    "S2"
)

s3_token_index = build_token_index(
    s3,
    s3_token_counts,
    "S3"
)


# =========================================================
# 6. GROUND TRUTH
# =========================================================

print("\nPreparing ground truth...")

gt_dict = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"].fillna("")
    )
)

print("Ground truth ready.")


# =========================================================
# 7. CANDIDATE GENERATION
# =========================================================

print("\nGenerating token candidates...")

total_true_matches = 0
recovered_true_matches = 0

total_candidates = 0
max_candidates = 0

num_s1 = len(s1)

for i, row in enumerate(s1.itertuples(index=False), start=1):

    candidates = set()

    text = f"{row.norm_name} {row.norm_address}"

    tokens = set(get_tokens(text))

    for token in tokens:

        key = (row.country, token)

        candidates.update(
            s2_token_index.get(key, [])
        )

        candidates.update(
            s3_token_index.get(key, [])
        )

    candidate_count = len(candidates)

    total_candidates += candidate_count

    max_candidates = max(
        max_candidates,
        candidate_count
    )

    # -----------------------------------------------------
    # Compare with ground truth
    # -----------------------------------------------------

    true_matches = gt_dict.get(
        row.entity_id,
        ""
    )

    if true_matches:

        true_ids = set(
            true_matches.split(",")
        )

        total_true_matches += len(true_ids)

        recovered_true_matches += len(
            true_ids.intersection(candidates)
        )

    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    if i % 100000 == 0:

        recall = (
            recovered_true_matches /
            total_true_matches
            if total_true_matches
            else 0
        )

        print(
            f"Processed {i:,} / {num_s1:,} | "
            f"Candidates: {total_candidates:,} | "
            f"Recall: {recall * 100:.2f}%"
        )


# =========================================================
# 8. RESULTS
# =========================================================

print("\n" + "=" * 70)
print("BLOCKING V2 RESULTS")
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
    recovered_true_matches /
    total_true_matches
    if total_true_matches
    else 0
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