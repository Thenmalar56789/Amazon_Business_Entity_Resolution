import pandas as pd
from collections import defaultdict, Counter
from rapidfuzz.fuzz import ratio, token_set_ratio


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

# Keep this many candidates per S1 after scoring
TOP_K = 20


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


def tokens(text):
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
# STORE S2/S3 RECORDS IN DICTIONARY
# ============================================================

print("Building record dictionaries...")

records = {}

for _, row in s2.iterrows():

    records[row["entity_id"]] = {
        "name": row["name_norm"],
        "address": row["address_norm"],
        "country": row["country_norm"],
        "source": "S2"
    }

for _, row in s3.iterrows():

    records[row["entity_id"]] = {
        "name": row["name_norm"],
        "address": row["address_norm"],
        "country": row["country_norm"],
        "source": "S3"
    }

print("Record dictionaries ready.")


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
# TOKEN FREQUENCY
# ============================================================

print("Counting token frequencies...")

name_freq = Counter()
address_freq = Counter()

for entity_id, record in records.items():

    country = record["country"]

    for token in tokens(record["name"]):
        name_freq[(country, token)] += 1

    for token in tokens(record["address"]):
        address_freq[(country, token)] += 1


# ============================================================
# TOKEN INDEXES
# ============================================================

print("Building token indexes...")

name_token_index = defaultdict(list)
address_token_index = defaultdict(list)

for entity_id, record in records.items():

    country = record["country"]

    for token in tokens(record["name"]):

        key = (country, token)

        if name_freq[key] <= MAX_TOKEN_FREQ:
            name_token_index[key].append(entity_id)

    for token in tokens(record["address"]):

        key = (country, token)

        if address_freq[key] <= MAX_TOKEN_FREQ:
            address_token_index[key].append(entity_id)


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
    len(x) for x in ground_truth.values()
)

print(
    f"Total true matches: {total_true_matches:,}"
)


# ============================================================
# SCORING FUNCTION
# ============================================================

def score_pair(source1, candidate):

    name1 = source1["name"]
    name2 = candidate["name"]

    addr1 = source1["address"]
    addr2 = candidate["address"]

    # Name similarities
    name_ratio = (
        ratio(name1, name2) / 100
        if name1 and name2 else 0
    )

    name_token_ratio = (
        token_set_ratio(name1, name2) / 100
        if name1 and name2 else 0
    )

    # Address similarities
    addr_ratio = (
        ratio(addr1, addr2) / 100
        if addr1 and addr2 else 0
    )

    addr_token_ratio = (
        token_set_ratio(addr1, addr2) / 100
        if addr1 and addr2 else 0
    )

    # Exact matches
    exact_name = int(
        bool(name1) and
        name1 == name2
    )

    exact_address = int(
        bool(addr1) and
        addr1 == addr2
    )

    # Token overlap
    name_tokens_1 = tokens(name1)
    name_tokens_2 = tokens(name2)

    addr_tokens_1 = tokens(addr1)
    addr_tokens_2 = tokens(addr2)

    if name_tokens_1 or name_tokens_2:

        name_union = (
            name_tokens_1 | name_tokens_2
        )

        name_intersection = (
            name_tokens_1 & name_tokens_2
        )

        name_jaccard = (
            len(name_intersection)
            / len(name_union)
            if name_union else 0
        )

    else:
        name_jaccard = 0

    if addr_tokens_1 or addr_tokens_2:

        addr_union = (
            addr_tokens_1 | addr_tokens_2
        )

        addr_intersection = (
            addr_tokens_1 & addr_tokens_2
        )

        addr_jaccard = (
            len(addr_intersection)
            / len(addr_union)
            if addr_union else 0
        )

    else:
        addr_jaccard = 0

    # Combined score
    score = (
        0.30 * name_ratio +
        0.20 * name_token_ratio +
        0.20 * addr_ratio +
        0.15 * addr_token_ratio +
        0.10 * name_jaccard +
        0.05 * addr_jaccard
    )

    # Strong bonus for exact fields
    if exact_name:
        score += 0.30

    if exact_address:
        score += 0.30

    return score


# ============================================================
# CANDIDATE GENERATION + TOP-K
# ============================================================

print("\nGenerating candidates and scoring...")

recovered_matches = 0
total_candidates = 0

processed = 0


for _, row in s1.iterrows():

    source1_id = row["entity_id"]
    country = row["country_norm"]

    source1 = {
        "name": row["name_norm"],
        "address": row["address_norm"],
        "country": country
    }

    candidates = set()

    # --------------------------------------------------------
    # Exact name
    # --------------------------------------------------------

    if source1["name"]:

        candidates.update(
            name_index.get(
                (country, source1["name"]),
                []
            )
        )

    # --------------------------------------------------------
    # Exact address
    # --------------------------------------------------------

    if source1["address"]:

        candidates.update(
            address_index.get(
                (country, source1["address"]),
                []
            )
        )

    # --------------------------------------------------------
    # Name token candidates
    # --------------------------------------------------------

    for token in tokens(source1["name"]):

        candidates.update(
            name_token_index.get(
                (country, token),
                []
            )
        )

    # --------------------------------------------------------
    # Address token candidates
    # --------------------------------------------------------

    for token in tokens(source1["address"]):

        candidates.update(
            address_token_index.get(
                (country, token),
                []
            )
        )

    total_candidates += len(candidates)

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    scored = []

    for entity_id in candidates:

        candidate = records[entity_id]

        score = score_pair(
            source1,
            candidate
        )

        scored.append(
            (score, entity_id)
        )

    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    scored.sort(
        reverse=True
    )

    top_candidates = scored[:TOP_K]

    top_ids = {
        entity_id
        for _, entity_id in top_candidates
    }

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    true_matches = ground_truth.get(
        source1_id,
        set()
    )

    recovered_matches += len(
        true_matches.intersection(top_ids)
    )

    processed += 1

    if processed % 10_000 == 0:

        recall = (
            recovered_matches
            / total_true_matches
            * 100
        )

        print(
            f"Processed {processed:,} / {len(s1):,}"
            f" | Raw candidates: {total_candidates:,}"
            f" | Top-{TOP_K} recall: {recall:.2f}%"
        )


# ============================================================
# RESULTS
# ============================================================

recall = (
    recovered_matches
    / total_true_matches
    * 100
)

avg_raw_candidates = (
    total_candidates
    / len(s1)
)

print("\n" + "=" * 70)
print("MATCHING V1 RESULTS")
print("=" * 70)

print(
    f"Total true matches: "
    f"{total_true_matches:,}"
)

print(
    f"Recovered after TOP-{TOP_K}: "
    f"{recovered_matches:,}"
)

print(
    f"Top-{TOP_K} recall: "
    f"{recall:.4f}%"
)

print(
    f"Average raw candidates/S1: "
    f"{avg_raw_candidates:.2f}"
)

print("\nDONE")