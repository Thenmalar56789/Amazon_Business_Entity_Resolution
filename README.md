# Amazon Business Entity Resolution

A Python-based entity-resolution pipeline developed for the Amazon ML Challenge 2026 to identify potential matching business records across large datasets.

## Overview

Business entity resolution identifies records that refer to the same real-world entity despite differences in their text fields or other attributes.

This project explores record matching using text normalization, candidate generation, blocking strategies, and similarity-based scoring. It focuses on reducing unnecessary comparisons while identifying potential matches across large datasets.

## Key Features

- **Text Normalization:** Standardizes text fields to improve matching consistency.
- **Exact-Match Indexing:** Uses matching keys to identify potential record matches efficiently.
- **Blocking Strategies:** Generates candidate pairs while reducing the number of comparisons required.
- **Candidate Generation:** Identifies records that are potentially related before detailed scoring.
- **Similarity Scoring:** Applies weighted similarity techniques to rank candidate matches.
- **Data Inspection:** Includes scripts for inspecting datasets, evaluating matches, and checking matching consistency.

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas
- **Techniques:** Entity Resolution, Text Normalization, Blocking, Candidate Generation, Similarity Scoring
- **Tools:** Git, GitHub, VS Code

## Approach

The pipeline follows these general stages:

1. **Data Inspection:** Examine available fields and understand the structure of the input records.
2. **Text Normalization:** Standardize text values to improve consistency.
3. **Indexing and Blocking:** Identify candidate records using matching keys and blocking strategies.
4. **Candidate Generation:** Generate potential record pairs for further comparison.
5. **Similarity Scoring:** Calculate matching scores using selected record attributes.
6. **Validation:** Inspect candidate matches and check consistency using evaluation scripts.

## Project Structure

```text
student_resource/
├── code/
│   └── business_entity_resolution/
│       ├── README.md
│       ├── requirements.txt
│       └── src/
│           └── final_submission.py
├── utils/
│   └── validate_submission.py
├── analyze_ground_truth.py
├── blocking_v1a.py
├── blocking_v1b.py
├── blocking_v2.py
├── blocking_v3.py
├── blocking_v4.py
├── blocking_v5.py
├── check_country_consistency.py
├── check_match_uniqueness.py
├── fast_final.py
├── final_submission.py
├── inspect_dataset.py
├── inspect_matches.py
├── matching_v1.py
├── matching_v2.py
├── requirements.txt
├── README.md
└── Documentation_template.md
```

*Note: The structure above reflects the submitted repository. Some scripts represent different iterations and experiments.*

## Getting Started

### Prerequisites

- Python 3.10 or a compatible version supported by the dependencies
- pip

### Installation

Clone the repository:

```bash
git clone https://github.com/Thenmalar56789/student_resource.git
cd student_resource
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

If the root requirements file does not contain all dependencies needed by the final implementation, install the dependencies listed in `code/business_entity_resolution/requirements.txt` as well.

### Running the Solution

Review the command-line arguments and input-file requirements in the final implementation before running it.

The primary implementation is located at:

```text
code/business_entity_resolution/src/final_submission.py
```

The repository also contains supporting scripts for dataset inspection, blocking experiments, matching, and validation.

**Note:** The original competition datasets are not included in this public repository. Provide the required input files separately in accordance with the competition rules.

## Dataset Scale

The project involved datasets containing approximately:

- 1.7 million records
- 4.9 million records
- 5.1 million records

These figures describe the datasets involved in the project. Processing capacity, runtime, matching accuracy, and evaluation performance depend on the specific implementation and test conditions.

## Learning Outcomes

- Applying entity-resolution techniques to large datasets
- Designing candidate-generation and blocking strategies
- Improving matching efficiency through indexing
- Developing data-processing pipelines in Python
- Comparing algorithmic approaches through iterative experimentation
- Validating record-matching results

## Disclaimer

This repository contains code developed as part of a learning and competition project for the Amazon ML Challenge 2026.

Amazon and its associated trademarks belong to their respective owners. This project is not affiliated with or endorsed by Amazon.

The repository does not claim a particular competition ranking or matching accuracy unless explicitly supported by verified evaluation results.
