# Business Entity Resolution - Amazon ML Challenge 2026

## Overview

This project implements a scalable business entity resolution pipeline for matching entities from Source 1 with corresponding entities in Source 2 and Source 3.

The pipeline performs:

1. Data loading
2. Text normalization
3. Country-aware blocking
4. Candidate generation
5. Candidate scoring
6. Precision-oriented match selection
7. Submission file generation

## Input

The inference pipeline expects the challenge dataset in the following structure:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── ground_truth.tsv
│
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv