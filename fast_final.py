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

TEST_S1 = "dataset/test/source1.csv"
TEST_S2 = "dataset/test/source2.csv"
TEST_S3 = "dataset/test/source3.csv"

OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MIN_TOKEN_LEN = 4
MAX_TOKEN_FREQ = 200

# Maximum final candidates retained per S1
TOP_K = 15


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
# LOAD TRAINING DATA
# ============================================================

print("Loading training data...")

s1_train = pd.read_csv(TRAIN_S1, sep="\t")
s2 = pd.read_csv(TRAIN_S2, sep="\t")
s3 = pd.read_csv(TRAIN_S3, sep="\t")
gt = pd.read_csv(GROUND_TRUTH, sep="\t")

print("Training data loaded.")


# ============================================================
# NORMALIZE TRAINING S2/S3
# ============================================================

print("Normalizing S2/S3...")

for df in [s2, s3]:

    df["name_norm"] = normalize(df["business_name"])
    df["address_norm"] = normalize(df["business_address"])
    df["country_norm"] = normalize(df["country"])


# ============================================================
# BUILD RECORD DICTIONARY
# ============================================================

print("Building S2/S3 records...")

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
# TRAINING VALIDATION
# ============================================================

print("\nTesting fast matcher on training data...")


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

recovered = 0
processed = 0


for _, row in s1_train.iterrows():

    s1_id = row["entity_id"]
    country = normalize(
        pd.Series([row["country"]])
    ).iloc[0]

    name = normalize(
        pd.Series([row["business_name"]])
    ).iloc[0]

    address = normalize(
        pd.Series([row["business_address"]])
    ).iloc[0]

    candidates = {}

    # --------------------------------------------------------
    # Exact name
    # --------------------------------------------------------

    for entity_id in name_index.get(
        (country, name), []
    ):
        candidates.setdefault(
            entity_id,
            [0, 0, 0, 0]
        )[0] += 100


    # --------------------------------------------------------
    # Exact address
    # --------------------------------------------------------

    for entity_id in address_index.get(
        (country, address), []
    ):
        candidates.setdefault(
            entity_id,
            [0, 0, 0, 0]
        )[1] += 100


    # --------------------------------------------------------
    # Name tokens
    # --------------------------------------------------------

    for token in get_tokens(name):

        for entity_id in name_token_index.get(
            (country, token), []
        ):

            candidates.setdefault(
                entity_id,
                [0, 0, 0, 0]
            )[2] += 1


    # --------------------------------------------------------
    # Address tokens
    # --------------------------------------------------------

    for token in get_tokens(address):

        for entity_id in address_token_index.get(
            (country, token), []
        ):

            candidates.setdefault(
                entity_id,
                [0, 0, 0, 0]
            )[3] += 1


    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    scored = []

    for entity_id, values in candidates.items():

        exact_name = values[0]
        exact_address = values[1]
        name_hits = values[2]
        address_hits = values[3]

        score = (
            exact_name
            + exact_address
            + name_hits * 10
            + address_hits * 8
        )

        # Strong combined evidence
        if name_hits >= 1 and address_hits >= 1:
            score += 15

        scored.append(
            (score, entity_id)
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
        s1_id,
        set()
    )

    recovered += len(
        true_matches.intersection(top_ids)
    )

    processed += 1

    if processed % 100_000 == 0:

        recall = (
            recovered
            / total_true_matches
            * 100
        )

        print(
            f"Validated "
            f"{processed:,}/{len(s1_train):,}"
            f" | Recall: {recall:.2f}%"
        )


print("\n" + "=" * 70)
print("FAST MATCHER TRAINING RESULT")
print("=" * 70)

print(
    f"Total true matches: "
    f"{total_true_matches:,}"
)

print(
    f"Recovered in TOP-{TOP_K}: "
    f"{recovered:,}"
)

print(
    f"Candidate recall: "
    f"{recovered / total_true_matches * 100:.4f}%"
)

print("\nDONE")