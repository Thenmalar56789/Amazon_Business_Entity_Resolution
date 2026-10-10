import pandas as pd
from collections import defaultdict, Counter


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_S1 = "dataset/train/train_source1.tsv"
TRAIN_S2 = "dataset/train/train_source2.tsv"
TRAIN_S3 = "dataset/train/train_source3.tsv"
GROUND_TRUTH = "dataset/train/train_ground_truth.tsv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text_vectorized(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def get_tokens(text):
    return {
        token for token in text.split()
        if len(token) >= 4
    }


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

s1 = pd.read_csv(TRAIN_S1, sep="\t")
s2 = pd.read_csv(TRAIN_S2, sep="\t")
s3 = pd.read_csv(TRAIN_S3, sep="\t")
gt = pd.read_csv(GROUND_TRUTH, sep="\t")

print("Loaded.")


# ============================================================
# NORMALIZATION
# ============================================================

print("Normalizing...")

for df in [s1, s2, s3]:
    df["name_norm"] = normalize_text_vectorized(
        df["business_name"]
    )

    df["address_norm"] = normalize_text_vectorized(
        df["business_address"]
    )

    df["country_norm"] = normalize_text_vectorized(
        df["country"]
    )

print("Normalization complete.")


# ============================================================
# EXACT INDEXES
# ============================================================

print("Building exact indexes...")

name_index_s2 = defaultdict(list)
addr_index_s2 = defaultdict(list)

name_index_s3 = defaultdict(list)
addr_index_s3 = defaultdict(list)


for _, row in s2.iterrows():

    country = row["country_norm"]

    if row["name_norm"]:
        name_index_s2[
            (country, row["name_norm"])
        ].append(row["entity_id"])

    if row["address_norm"]:
        addr_index_s2[
            (country, row["address_norm"])
        ].append(row["entity_id"])


for _, row in s3.iterrows():

    country = row["country_norm"]

    if row["name_norm"]:
        name_index_s3[
            (country, row["name_norm"])
        ].append(row["entity_id"])

    if row["address_norm"]:
        addr_index_s3[
            (country, row["address_norm"])
        ].append(row["entity_id"])


# ============================================================
# TOKEN FREQUENCIES
# ============================================================

print("Counting token frequencies...")

name_token_freq_s2 = Counter()
addr_token_freq_s2 = Counter()

name_token_freq_s3 = Counter()
addr_token_freq_s3 = Counter()


for i, row in s2.iterrows():

    country = row["country_norm"]

    for token in get_tokens(row["name_norm"]):
        name_token_freq_s2[(country, token)] += 1

    for token in get_tokens(row["address_norm"]):
        addr_token_freq_s2[(country, token)] += 1

    if (i + 1) % 1_000_000 == 0:
        print(
            f"S2: {i + 1:,} / {len(s2):,}"
        )


for i, row in s3.iterrows():

    country = row["country_norm"]

    for token in get_tokens(row["name_norm"]):
        name_token_freq_s3[(country, token)] += 1

    for token in get_tokens(row["address_norm"]):
        addr_token_freq_s3[(country, token)] += 1

    if (i + 1) % 1_000_000 == 0:
        print(
            f"S3: {i + 1:,} / {len(s3):,}"
        )


# ============================================================
# TOKEN INDEXES
# ============================================================

print("Building token indexes...")

name_token_index_s2 = defaultdict(list)
addr_token_index_s2 = defaultdict(list)

name_token_index_s3 = defaultdict(list)
addr_token_index_s3 = defaultdict(list)

MAX_FREQ = 200


for i, row in s2.iterrows():

    country = row["country_norm"]

    for token in get_tokens(row["name_norm"]):

        key = (country, token)

        if name_token_freq_s2[key] <= MAX_FREQ:
            name_token_index_s2[key].append(
                row["entity_id"]
            )

    for token in get_tokens(row["address_norm"]):

        key = (country, token)

        if addr_token_freq_s2[key] <= MAX_FREQ:
            addr_token_index_s2[key].append(
                row["entity_id"]
            )

    if (i + 1) % 1_000_000 == 0:
        print(
            f"S2 index: {i + 1:,} / {len(s2):,}"
        )


for i, row in s3.iterrows():

    country = row["country_norm"]

    for token in get_tokens(row["name_norm"]):

        key = (country, token)

        if name_token_freq_s3[key] <= MAX_FREQ:
            name_token_index_s3[key].append(
                row["entity_id"]
            )

    for token in get_tokens(row["address_norm"]):

        key = (country, token)

        if addr_token_freq_s3[key] <= MAX_FREQ:
            addr_token_index_s3[key].append(
                row["entity_id"]
            )

    if (i + 1) % 1_000_000 == 0:
        print(
            f"S3 index: {i + 1:,} / {len(s3):,}"
        )


# ============================================================
# GROUND TRUTH
# ============================================================

print("Preparing ground truth...")

ground_truth = {}

for _, row in gt.iterrows():

    source1_id = row["source1_entity_id"]
    value = row["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":
        ground_truth[source1_id] = set()
    else:
        ground_truth[source1_id] = set(
            str(value).split(",")
        )

print("Ground truth ready.")


# ============================================================
# CANDIDATE GENERATION
# ============================================================

print("\nGenerating HYBRID V4 candidates...")

total_true_matches = sum(
    len(matches)
    for matches in ground_truth.values()
)

recovered_true_matches = 0
total_candidates = 0
max_candidates = 0


for idx, row in s1.iterrows():

    country = row["country_norm"]

    candidates = set()

    # --------------------------------------------------------
    # 1. EXACT NAME
    # --------------------------------------------------------

    if row["name_norm"]:

        key = (
            country,
            row["name_norm"]
        )

        candidates.update(
            name_index_s2.get(key, [])
        )

        candidates.update(
            name_index_s3.get(key, [])
        )

    # --------------------------------------------------------
    # 2. EXACT ADDRESS
    # --------------------------------------------------------

    if row["address_norm"]:

        key = (
            country,
            row["address_norm"]
        )

        candidates.update(
            addr_index_s2.get(key, [])
        )

        candidates.update(
            addr_index_s3.get(key, [])
        )

    # --------------------------------------------------------
    # 3. RARE NAME TOKENS
    # --------------------------------------------------------

    for token in get_tokens(row["name_norm"]):

        key = (
            country,
            token
        )

        candidates.update(
            name_token_index_s2.get(key, [])
        )

        candidates.update(
            name_token_index_s3.get(key, [])
        )

    # --------------------------------------------------------
    # 4. RARE ADDRESS TOKENS
    # --------------------------------------------------------

    for token in get_tokens(row["address_norm"]):

        key = (
            country,
            token
        )

        candidates.update(
            addr_token_index_s2.get(key, [])
        )

        candidates.update(
            addr_token_index_s3.get(key, [])
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total_candidates += len(candidates)

    max_candidates = max(
        max_candidates,
        len(candidates)
    )

    true_matches = ground_truth.get(
        row["entity_id"],
        set()
    )

    recovered_true_matches += len(
        true_matches.intersection(candidates)
    )

    if (idx + 1) % 100_000 == 0:

        recall = (
            recovered_true_matches
            / total_true_matches
            * 100
        )

        print(
            f"Processed {idx + 1:,} / {len(s1):,}"
            f" | Candidates: {total_candidates:,}"
            f" | Recall: {recall:.2f}%"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

recall = (
    recovered_true_matches
    / total_true_matches
    * 100
)

avg_candidates = (
    total_candidates
    / len(s1)
)


print("\n" + "=" * 70)
print("BLOCKING V4 RESULTS")
print("=" * 70)

print(
    f"Total true matches: "
    f"{total_true_matches:,}"
)

print(
    f"Recovered true matches: "
    f"{recovered_true_matches:,}"
)

print(
    f"Candidate recall: "
    f"{recall:.4f}%"
)

print(
    f"Total generated candidates: "
    f"{total_candidates:,}"
)

print(
    f"Average candidates per S1: "
    f"{avg_candidates:.2f}"
)

print(
    f"Maximum candidates for one S1: "
    f"{max_candidates:,}"
)

print("\nDONE")