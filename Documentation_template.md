# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your actual team name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We developed a scalable business entity resolution pipeline to identify matching business entities across three independent data sources. The solution combines text normalization, country-aware blocking, exact and token-based candidate generation, lightweight similarity scoring, and precision-oriented match selection.

The approach was designed to handle millions of records efficiently while accounting for noisy business names, address variations, missing fields, spelling errors, and multilingual representations.

---

## 2. Methodology

### 2.1 Problem Analysis

The training data contains business entities from Source 1, Source 2, and Source 3. Source 1 acts as the reference source, and each Source 1 entity can have zero, one, or multiple corresponding entities in Source 2 and Source 3.

Exploratory analysis showed:

- Business names contain spelling errors, reordered words, additional words, and multilingual variations.
- Addresses contain spelling variations, reordered components, and formatting differences.
- Some Source 2 and Source 3 records have missing business names or addresses.
- France appears in the test data, so country was treated as an open-set value rather than hard-coding specific countries.
- The number of true matches varies substantially across Source 1 entities, confirming that this is a zero-to-many matching problem rather than a top-1 classification problem.
- Approximately 5.6% of training Source 1 entities have no matches.
- Approximately 80.5% have matches in both Source 2 and Source 3.
- Analysis of the training ground truth showed that all observed true matches belong to the same country as the corresponding Source 1 entity.

These observations motivated a country-aware, precision-focused candidate-generation and matching strategy.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Similarity-based Entity Resolution

**Core Approach:** Reduce the large search space using country-aware blocking and exact/token-based matching signals, followed by lightweight candidate scoring and precision-oriented selection.

The pipeline consists of:

1. Text normalization
2. Country-aware blocking
3. Candidate generation
4. Candidate scoring
5. Match selection
6. Submission generation

---

## 3. Candidate Generation (Blocking)

Since the datasets contain millions of records, exhaustive comparison between every Source 1 entity and every Source 2/Source 3 entity is computationally infeasible.

The solution therefore uses blocking to substantially reduce the comparison space.

### Blocking keys used

- Country
- Exact normalized business name
- Exact normalized business address
- Selective token overlap in business names and addresses
- Frequency restrictions on common tokens to avoid excessive candidate generation

Country was used as a strong blocking signal because the training ground truth showed 100% country consistency among observed true matches.

### Candidate generation experiments

Several blocking strategies were evaluated during development:

- Exact normalized name/address blocking
- Token-based blocking
- Combined exact and selective token blocking
- Rare-token-based blocking
- More aggressive multi-token blocking

The final approach was selected based on the trade-off between candidate recall, candidate-set size, and computational cost.

The final test inference generated:

- **1,732,544 Source 1 entities**
- **1,629,449 Source 1 entities with non-empty candidate sets**
- **103,095 Source 1 entities with empty candidate sets**

The candidate set was restricted to a maximum number of high-ranking candidates per Source 1 entity before final matching.

### Preventing loss of true matches

Multiple complementary blocking signals were used rather than relying on a single field. This was important because names and addresses can contain spelling errors, missing components, reordered words, and other noise.

---

## 4. Matching Model

The final solution uses a lightweight similarity-based matching pipeline rather than a separately trained neural network or gradient-boosting classifier.

### Features used

**Name features:**
- Normalized business-name equality
- Token overlap
- Text similarity between candidate names

**Address features:**
- Normalized address equality
- Token overlap
- Text similarity between candidate addresses

**Other features:**
- Country agreement
- Combined name/address evidence
- Exact normalized name and address matches

Candidates receiving strong evidence from exact normalized fields were prioritized. Additional similarity evidence was used for noisy records.

### Model type

**Model:** Rule-based / similarity-based entity resolution pipeline with candidate ranking.

No external entity database, external API, geocoding service, or external data augmentation was used.

### Threshold / selection strategy

The final matching decision uses precision-oriented acceptance criteria because the challenge metric, F0.5, gives greater importance to precision.

Candidates are accepted when they satisfy strong matching conditions based on normalized name, normalized address, or sufficient combined similarity evidence.

Multiple candidates can be selected for a Source 1 entity because the ground truth permits multiple true matches.

---

## 5. Results & Error Analysis

### Development experiments

Several approaches were tested on the training data.

The experiments showed that:

- Exact matching alone was fast but missed many noisy matches.
- Basic token blocking increased recall but produced very large candidate sets.
- Combined blocking provided a better balance between candidate recall and computational cost.
- Fuzzy matching using RapidFuzz was tested, but its computational cost was high at the scale of millions of records.
- The final approach therefore emphasized efficient blocking and lightweight candidate scoring.

### Final output

The final inference generated:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

The official challenge validator was run successfully.

The strong validation check also confirmed that:

- Every required Source 1 test entity is present.
- Output row counts are correct.
- Returned Source 2/Source 3 IDs are valid.
- Final matches are contained within the generated candidate sets.
- No blocking validation errors were detected.

### Error analysis

Potential false positives arise mainly from businesses with similar or generic names, while potential false negatives arise from severe spelling variations, missing fields, multilingual representations, or records that do not survive candidate generation.

The precision-oriented matching strategy was designed to reduce incorrect entity merges while retaining useful coverage.

---

## 6. Conclusion

The final solution provides a scalable approach to business entity resolution using text normalization, country-aware blocking, exact and token-based candidate generation, lightweight similarity scoring, and precision-oriented match selection.

The experiments demonstrated the importance of blocking for handling millions of records efficiently and showed that combining multiple complementary signals is necessary to handle noisy business names and addresses.

---

## Appendix

### A. Code Artefacts

The submission package contains the complete inference code under:

`code/business_entity_resolution/`

The structure is:

```text
code/
└── business_entity_resolution/
    ├── src/
    │   └── final_submission.py
    ├── README.md
    └── requirements.txt