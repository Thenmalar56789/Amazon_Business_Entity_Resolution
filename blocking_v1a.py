import pandas as pd
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

BASE = Path("dataset/train")


# ---------------------------------------------------------
# 1. NORMALIZATION
# ---------------------------------------------------------

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9\u0900-\u097f]+", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

print("Loading datasets...")

s1 = pd.read_csv(
    BASE / "train_source1.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"]
)

s2 = pd.read_csv(
    BASE / "train_source2.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"]
)

s3 = pd.read_csv(
    BASE / "train_source3.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"]
)

gt = pd.read_csv(
    BASE / "train_ground_truth.tsv",
    sep="\t"
)

print("Loaded.")


# ---------------------------------------------------------
# 3. NORMALIZE NAME + ADDRESS
# ---------------------------------------------------------

print("Normalizing names and addresses...")

for df in [s1, s2, s3]:

    df["norm_name"] = df["business_name"].map(normalize_text)
    df["norm_address"] = df["business_address"].map(normalize_text)

print("Normalization complete.")


# ---------------------------------------------------------
# 4. BUILD BLOCK INDEXES
# ---------------------------------------------------------

print("Building indexes...")

name_index_s2 = defaultdict(list)
address_index_s2 = defaultdict(list)

name_index_s3 = defaultdict(list)
address_index_s3 = defaultdict(list)


# S2 indexes
for row in s2.itertuples(index=False):

    if row.norm_name:
        key = (row.country, row.norm_name)
        name_index_s2[key].append(row.entity_id)

    if row.norm_address:
        key = (row.country, row.norm_address)
        address_index_s2[key].append(row.entity_id)


# S3 indexes
for row in s3.itertuples(index=False):

    if row.norm_name:
        key = (row.country, row.norm_name)
        name_index_s3[key].append(row.entity_id)

    if row.norm_address:
        key = (row.country, row.norm_address)
        address_index_s3[key].append(row.entity_id)


print("Indexes built.")


# ---------------------------------------------------------
# 5. CREATE CANDIDATES
# ---------------------------------------------------------

print("Generating candidates...")

candidate_counts = []

total_true_matches = 0
recovered_true_matches = 0

# Ground truth lookup
gt_dict = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"].fillna("")
    )
)


for i, row in enumerate(s1.itertuples(index=False)):

    candidates = set()

    # -----------------------------------------
    # Exact normalized NAME block
    # -----------------------------------------

    if row.norm_name:

        key = (row.country, row.norm_name)

        candidates.update(name_index_s2.get(key, []))
        candidates.update(name_index_s3.get(key, []))


    # -----------------------------------------
    # Exact normalized ADDRESS block
    # -----------------------------------------

    if row.norm_address:

        key = (row.country, row.norm_address)

        candidates.update(address_index_s2.get(key, []))
        candidates.update(address_index_s3.get(key, []))


    candidate_counts.append(len(candidates))


    # -----------------------------------------
    # Candidate recall
    # -----------------------------------------

    true_matches = gt_dict.get(row.entity_id, "")

    if true_matches:

        true_ids = set(true_matches.split(","))

        total_true_matches += len(true_ids)

        recovered_true_matches += len(
            true_ids.intersection(candidates)
        )


    # Progress
    if (i + 1) % 100000 == 0:

        print(
            f"Processed {i + 1:,} / {len(s1):,}"
        )


# ---------------------------------------------------------
# 6. RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BLOCKING V1A RESULTS")
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

print(
    "Candidate recall:",
    round(recall * 100, 4),
    "%"
)

print(
    "Average candidates per S1:",
    round(sum(candidate_counts) / len(candidate_counts), 2)
)

print(
    "Maximum candidates for one S1:",
    max(candidate_counts)
)

print(
    "S1 entities:",
    len(candidate_counts)
)

print("\nDONE")