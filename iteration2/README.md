# Iteration 2: The Math Foundation

This folder contains the Iteration 2 deliverables: the data analysis, the simulation check and the report.

## Contents

```
iteration2/
├── README.md
├── src/
│   └── data_analysis.ipynb        data loading, cleaning, description and simulation check
├── data/
│   └── SMSSpamCollection          raw data (see "Getting the data")
├── figures/
│   ├── class_balance.png
│   ├── message_length_by_class.png
│   └── class_similarity_accuracy.png
└── docs/
    ├── iteration2_report_updated_.pdf      Overleaf report
    └── DS5020_Project_Plan.xlsx   task table, timeline and contributions
```

## What the notebook does

`src/data_analysis.ipynb` has two parts.

**Part 1: Real SMS data**
- Loads all 5,574 messages, removes 414 repeated messages (5,160 remain) and records each cleaning step.
- Describes the data: class balance (87.6% ham, 12.4% spam) and message length by class.
- Splits the data with a stratified 80/20 split written in NumPy (seed 0): 4,128 training and 1,032 test messages.
- Builds the vocabulary from the training set only (7,629 words) and the count matrix X.
- Checks Assumptions 3 to 6 of the report on the real data.

**Part 2: Simulation with known truth**
- Generates 2,000 messages from a two-class multinomial model (20 words, 3 words per message, priors 0.8/0.2, seed 0), as set in the Iteration 2 handout.
- Compares the true parameters with the estimates.
- Checks each derivation of the report numerically (Propositions 1 to 5).
- Runs a similarity experiment: as the two word distributions become more alike, balanced accuracy falls from 0.87 to 0.50.

## How to run

1. Put `SMSSpamCollection` in `iteration2/data/` (see the main README for the download link).
2. Install the requirements: `pip install numpy pandas matplotlib scipy`.
3. Open `iteration2/src/data_analysis.ipynb`. The notebook runs from inside `src/`, reads `../data/` and saves the figures to `../figures/`.
4. Restart the kernel and run all cells from top to bottom.

Only `numpy`, `pandas`, `matplotlib` and `scipy.stats` are used, as required for Iteration 2.

## Report

`docs/iteration2_report.pdf` is the Overleaf report. It contains the notation, definitions, assumptions, derivations (Propositions 1 to 5), the algorithm, the data description and the simulation check.

## Team

| Task | Owner | Checker |
|---|---|---|
| Notation, definitions, assumptions, Q1, Q2, Q4, algorithm | Debaswini Moharana | Keertana Srinivasan |
| Q3, Q5, data analysis, simulation check | Keertana Srinivasan | Debaswini Moharana |

Project leader for Iteration 2: Debaswini Moharana.
