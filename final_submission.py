import os
import pandas as pd
from collections import defaultdict, Counter


# ============================================================
# PATHS
# ============================================================

TEST_S1 = "dataset/test/test_source1.tsv"
TEST_S2 = "dataset/test/test_source2.tsv"
TEST_S3 = "dataset/test/test_source3.tsv"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MATCHING_OUT = os.path.join(
    OUTPUT_DIR, "matching_results.tsv"
)

CANDIDATE_OUT = os.path.join(
    OUTPUT_DIR, "candidate_pairs.tsv"
)


# ============================================================
# SETTINGS
# ============================================================

MIN_TOKEN_LEN = 4
MAX_TOKEN_FREQ = 200

# Same candidate limit as our validated fast matcher
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
# LOAD TEST DATA
# ============================================================

print("Loading test data...")

s1 = pd.read_csv(
    TEST_S1,
    sep="\t"
)

s2 = pd.read_csv(
    TEST_S2,
    sep="\t"
)

s3 = pd.read_csv(
    TEST_S3,
    sep="\t"
)

print("Test data loaded.")

print(f"S1: {len(s1):,}")
print(f"S2: {len(s2):,}")
print(f"S3: {len(s3):,}")


# ============================================================
# NORMALIZE S1
# ============================================================

print("Normalizing S1...")

s1["name_norm"] = normalize(
    s1["business_name"]
)

s1["address_norm"] = normalize(
    s1["business_address"]
)

s1["country_norm"] = normalize(
    s1["country"]
)


# ============================================================
# NORMALIZE S2/S3
# ============================================================

print("Normalizing S2/S3...")

for df in [s2, s3]:

    df["name_norm"] = normalize(
        df["business_name"]
    )

    df["address_norm"] = normalize(
        df["business_address"]
    )

    df["country_norm"] = normalize(
        df["country"]
    )

print("Normalization complete.")


# ============================================================
# BUILD RECORD DICTIONARY
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
            name_token_index[key].append(
                entity_id
            )

    for token in get_tokens(record["address"]):

        key = (country, token)

        if address_freq[key] <= MAX_TOKEN_FREQ:
            address_token_index[key].append(
                entity_id
            )

print("Indexes ready.")


# ============================================================
# PREPARE OUTPUT FILES
# ============================================================

print("Preparing output files...")

with open(
    MATCHING_OUT,
    "w",
    encoding="utf-8",
    newline=""
) as f_match, open(
    CANDIDATE_OUT,
    "w",
    encoding="utf-8",
    newline=""
) as f_candidate:

    # Headers
    f_match.write(
        "source1_entity_id\tmatched_entity_ids\n"
    )

    f_candidate.write(
        "source1_entity_id\tcandidate_entity_ids\n"
    )

    # ========================================================
    # PROCESS TEST S1
    # ========================================================

    total = len(s1)

    for i, row in enumerate(
        s1.itertuples(index=False),
        start=1
    ):

        s1_id = row.entity_id
        country = row.country_norm
        name = row.name_norm
        address = row.address_norm

        candidates = {}

        # ----------------------------------------------------
        # Exact name
        # ----------------------------------------------------

        if name:

            for entity_id in name_index.get(
                (country, name),
                []
            ):

                candidates.setdefault(
                    entity_id,
                    [0, 0, 0, 0]
                )[0] += 100

        # ----------------------------------------------------
        # Exact address
        # ----------------------------------------------------

        if address:

            for entity_id in address_index.get(
                (country, address),
                []
            ):

                candidates.setdefault(
                    entity_id,
                    [0, 0, 0, 0]
                )[1] += 100

        # ----------------------------------------------------
        # Name tokens
        # ----------------------------------------------------

        for token in get_tokens(name):

            for entity_id in name_token_index.get(
                (country, token),
                []
            ):

                candidates.setdefault(
                    entity_id,
                    [0, 0, 0, 0]
                )[2] += 1

        # ----------------------------------------------------
        # Address tokens
        # ----------------------------------------------------

        for token in get_tokens(address):

            for entity_id in address_token_index.get(
                (country, token),
                []
            ):

                candidates.setdefault(
                    entity_id,
                    [0, 0, 0, 0]
                )[3] += 1

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

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

            # Combined evidence
            if (
                name_hits >= 1
                and address_hits >= 1
            ):
                score += 15

            scored.append(
                (
                    score,
                    entity_id,
                    exact_name,
                    exact_address,
                    name_hits,
                    address_hits
                )
            )

        # Highest score first
        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        # ----------------------------------------------------
        # FINAL CANDIDATE SET
        # ----------------------------------------------------

        top_scored = scored[:TOP_K]

        candidate_ids = [
            x[1]
            for x in top_scored
        ]

        # Candidate file gets EXACTLY the candidates
        # that our final matching stage considers.
        f_candidate.write(
            s1_id
            + "\t"
            + ",".join(candidate_ids)
            + "\n"
        )

        # ----------------------------------------------------
        # PRECISION-ORIENTED MATCH SELECTION
        # ----------------------------------------------------

        matched_ids = []

        for (
            score,
            entity_id,
            exact_name,
            exact_address,
            name_hits,
            address_hits
        ) in top_scored:

            strong_match = False

            # Exact normalized name
            if exact_name >= 100:
                strong_match = True

            # Exact normalized address
            elif exact_address >= 100:
                strong_match = True

            # Strong combined token evidence
            elif score >= 30:
                strong_match = True

            if strong_match:
                matched_ids.append(
                    entity_id
                )

        # Remove duplicates while preserving order
        matched_ids = list(
            dict.fromkeys(matched_ids)
        )

        f_match.write(
            s1_id
            + "\t"
            + ",".join(matched_ids)
            + "\n"
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if i % 100_000 == 0:

            print(
                f"Processed "
                f"{i:,}/{total:,}"
            )


print("\n" + "=" * 70)
print("FINAL TEST INFERENCE COMPLETE")
print("=" * 70)

print(
    f"matching_results.tsv: {MATCHING_OUT}"
)

print(
    f"candidate_pairs.tsv: {CANDIDATE_OUT}"
)

print("DONE")