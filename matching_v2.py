import os
import pandas as pd
from collections import defaultdict, Counter


# ============================================================
# PATHS
# ============================================================

TRAIN_S1 = "dataset/train/train_source1.tsv"
TRAIN_S2 = "dataset/train/train_source2.tsv"
TRAIN_S3 = "dataset/train/train_source3.tsv"
GROUND_TRUTH = "dataset/train/train_ground_truth.tsv"


# ============================================================
# SETTINGS
# ============================================================

MIN_TOKEN_LEN = 4
MAX_TOKEN_FREQ = 200
TOP_K = 50


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(series):
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
        x for x in text.split()
        if len(x) >= MIN_TOKEN_LEN
    }


# ============================================================
# LOAD
# ============================================================

print("Loading datasets...")

s1 = pd.read_csv(TRAIN_S1, sep="\t")
s2 = pd.read_csv(TRAIN_S2, sep="\t")
s3 = pd.read_csv(TRAIN_S3, sep="\t")
gt = pd.read_csv(GROUND_TRUTH, sep="\t")

print("Loaded.")


# ============================================================
# NORMALIZE
# ============================================================

print("Normalizing...")

for df in [s1, s2, s3]:

    df["name_norm"] = normalize(df["business_name"])
    df["address_norm"] = normalize(df["business_address"])
    df["country_norm"] = normalize(df["country"])

print("Normalization complete.")


# ============================================================
# RECORDS
# ============================================================

print("Building records...")

records = {}

for _, row in s2.iterrows():

    records[row["entity_id"]] = {
        "name": row["name_norm"],
        "address": row["address_norm"],
        "country": row["country_norm"]
    }


for _, row in s3.iterrows():

    records[row["entity_id"]] = {
        "name": row["name_norm"],
        "address": row["address_norm"],
        "country": row["country_norm"]
    }

print("Records ready.")


# ============================================================
# EXACT INDEXES
# ============================================================

print("Building exact indexes...")

name_index = defaultdict(list)
address_index = defaultdict(list)

for entity_id, record in records.items():

    country = record["country"]

    if record["name"]:
        name_index[
            (country, record["name"])
        ].append(entity_id)

    if record["address"]:
        address_index[
            (country, record["address"])
        ].append(entity_id)


# ============================================================
# TOKEN FREQUENCIES
# ============================================================

print("Counting token frequencies...")

name_freq = Counter()
address_freq = Counter()

for entity_id, record in records.items():

    country = record["country"]

    for token in get_tokens(record["name"]):
        name_freq[(country, token)] += 1

    for token in get_tokens(record["address"]):
        address_freq[(country, token)] += 1


# ============================================================
# TOKEN INDEXES
# ============================================================

print("Building token indexes...")

name_token_index = defaultdict(list)
address_token_index = defaultdict(list)

for entity_id, record in records.items():

    country = record["country"]

    for token in get_tokens(record["name"]):

        key = (country, token)

        if name_freq[key] <= MAX_TOKEN_FREQ:
            name_token_index[key].append(entity_id)

    for token in get_tokens(record["address"]):

        key = (country, token)

        if address_freq[key] <= MAX_TOKEN_FREQ:
            address_token_index[key].append(entity_id)

print("Indexes ready.")


# ============================================================
# GROUND TRUTH
# ============================================================

print("Preparing ground truth...")

ground_truth = {}

for _, row in gt.iterrows():

    value = row["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":
        ground_truth[row["source1_entity_id"]] = set()

    else:
        ground_truth[row["source1_entity_id"]] = set(
            str(value).split(",")
        )

total_true_matches = sum(
    len(x)
    for x in ground_truth.values()
)

print(
    f"Total true matches: {total_true_matches:,}"
)


# ============================================================
# CANDIDATE SCORING
# ============================================================

def calculate_features(s1_name, s1_addr, candidate):

    c_name = candidate["name"]
    c_addr = candidate["address"]

    s1_name_tokens = get_tokens(s1_name)
    c_name_tokens = get_tokens(c_name)

    s1_addr_tokens = get_tokens(s1_addr)
    c_addr_tokens = get_tokens(c_addr)

    # ------------------------------
    # Exact signals
    # ------------------------------

    exact_name = int(
        bool(s1_name)
        and s1_name == c_name
    )

    exact_address = int(
        bool(s1_addr)
        and s1_addr == c_addr
    )

    # ------------------------------
    # Token intersections
    # ------------------------------

    name_shared = len(
        s1_name_tokens & c_name_tokens
    )

    address_shared = len(
        s1_addr_tokens & c_addr_tokens
    )

    # ------------------------------
    # Jaccard
    # ------------------------------

    name_union = (
        s1_name_tokens | c_name_tokens
    )

    address_union = (
        s1_addr_tokens | c_addr_tokens
    )

    name_jaccard = (
        name_shared / len(name_union)
        if name_union else 0
    )

    address_jaccard = (
        address_shared / len(address_union)
        if address_union else 0
    )

    # ------------------------------
    # Length similarity
    # ------------------------------

    name_length_difference = abs(
        len(s1_name) - len(c_name)
    )

    address_length_difference = abs(
        len(s1_addr) - len(c_addr)
    )

    # ------------------------------
    # Combined score
    # ------------------------------

    score = (
        exact_name * 1000
        + exact_address * 1000
        + name_shared * 20
        + address_shared * 15
        + name_jaccard * 100
        + address_jaccard * 80
    )

    # Strong combined signal
    if name_shared >= 1 and address_shared >= 1:
        score += 50

    if name_shared >= 2:
        score += 40

    if address_shared >= 2:
        score += 30

    return {
        "score": score,
        "exact_name": exact_name,
        "exact_address": exact_address,
        "name_shared": name_shared,
        "address_shared": address_shared,
        "name_jaccard": name_jaccard,
        "address_jaccard": address_jaccard,
        "name_length_difference": name_length_difference,
        "address_length_difference": address_length_difference
    }


# ============================================================
# VALIDATION
# ============================================================

print("\nScoring candidates...")

recovered = 0
processed = 0

for _, row in s1.iterrows():

    source1_id = row["entity_id"]
    country = row["country_norm"]

    name = row["name_norm"]
    address = row["address_norm"]

    candidates = set()

    # ========================================================
    # EXACT NAME
    # ========================================================

    if name:

        candidates.update(
            name_index.get(
                (country, name),
                []
            )
        )

    # ========================================================
    # EXACT ADDRESS
    # ========================================================

    if address:

        candidates.update(
            address_index.get(
                (country, address),
                []
            )
        )

    # ========================================================
    # NAME TOKEN BLOCKING
    # ========================================================

    for token in get_tokens(name):

        candidates.update(
            name_token_index.get(
                (country, token),
                []
            )
        )

    # ========================================================
    # ADDRESS TOKEN BLOCKING
    # ========================================================

    for token in get_tokens(address):

        candidates.update(
            address_token_index.get(
                (country, token),
                []
            )
        )

    # ========================================================
    # SCORE
    # ========================================================

    scored = []

    for entity_id in candidates:

        features = calculate_features(
            name,
            address,
            records[entity_id]
        )

        scored.append(
            (
                features["score"],
                entity_id
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    top_ids = {
        entity_id
        for _, entity_id in scored[:TOP_K]
    }

    true_matches = ground_truth.get(
        source1_id,
        set()
    )

    recovered += len(
        true_matches & top_ids
    )

    processed += 1

    if processed % 100_000 == 0:

        recall = (
            recovered
            / total_true_matches
            * 100
        )

        print(
            f"Processed "
            f"{processed:,}/{len(s1):,}"
            f" | TOP-{TOP_K} recall: "
            f"{recall:.2f}%"
        )


# ============================================================
# RESULT
# ============================================================

recall = (
    recovered
    / total_true_matches
    * 100
)

print("\n" + "=" * 70)
print("MATCHING V2 RESULTS")
print("=" * 70)

print(
    f"Total true matches: "
    f"{total_true_matches:,}"
)

print(
    f"Recovered after TOP-{TOP_K}: "
    f"{recovered:,}"
)

print(
    f"TOP-{TOP_K} recall: "
    f"{recall:.4f}%"
)

print("\nDONE")