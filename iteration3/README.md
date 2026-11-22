# Iteration 3: Small Worked Example, From-Scratch Implementation and Verification

This folder turns the Iteration 2 math into working code and verifies it three ways: on a tiny example solved with exact fractions, on simulated data where the truth is known, and on the real SMS data against scikit-learn.

## Contents

```
iteration3/
├── README.md
├── src/
│   ├── methods.py            one function per algorithm step (NumPy only)
│   ├── simulate.py           simulate_nb(...): the Iteration 2 simulation model
│   ├── small_example.py      Math Q1 and Q2: the tiny example step by step
│   ├── check_simulation.py   simulation test: truth, library, error vs n, stress test
│   └── run_real.py           real data: cleaning, library comparison, Q3, Q4, assumption checks
├── data/
│   └── SMSSpamCollection     
└── figures/
    ├── simulation_error_vs_n.png
    └── stress_test_precision_recall.png
```

## Requirements

Python 3 with `numpy`, `pandas`, `matplotlib`, `scipy` and `scikit-learn`:

```
pip install numpy pandas matplotlib scipy scikit-learn
```

The classifier in `methods.py` uses **NumPy only**. scikit-learn and SciPy are used only in the check scripts, to verify our results.

## Getting the data

Download the SMS Spam Collection from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/228/sms+spam+collection) and place the file `SMSSpamCollection` in `iteration3/data/`.

## How to run each script

Run from inside the `iteration3/` folder.

| Script | Command | What it does |
|---|---|---|
| Small example | `python src/small_example.py` | 5 messages, 5 words: priors, counts, smoothed likelihoods, scores with probabilities and with logs, posterior, all as exact fractions; second-way checks; comparison with `methods.py`; underflow demo (Math Q2) |
| Simulation test | `python src/check_simulation.py` | Iteration 2 simulation (seed 0): comparison with the true parameters and with `MultinomialNB`, error against n on log–log axes, assumption checks, stress test; saves both figures |
| Real data | `python src/run_real.py` | Cleans the data (5,574 → 5,160 messages), stratified 80/20 split (seed 0), compares with `MultinomialNB` (Math Q3), confusion matrix, precision and recall (Math Q4), assumption checks including conditional independence (Math Q5) |

## Key results

| Check | Result |
|---|---|
| Small example vs exact fractions | max difference ≤ 1.1 × 10⁻¹⁶ |
| Simulation: predictions vs `MultinomialNB` | 100% agree; log-probabilities within 3.6 × 10⁻¹⁵ |
| Simulation: error against the truth | log–log slope −0.504 (theory −0.5) |
| Real data: predictions vs `MultinomialNB` | 100% agree; log-probabilities within 8.5 × 10⁻¹⁴ |
| Real data: test set | accuracy 0.980, precision 0.957, recall 0.875 |

## Team

| Part | Owner | Checker |
|---|---|---|
| Small example, `small_example.py`, Math Q1, Q2, Q5 | Debaswini Moharana | Keertana Srinivasan |
| `methods.py`, `simulate.py`, `check_simulation.py`, `run_real.py`, Math Q3, Q4, verification table | Keertana Srinivasan | Debaswini Moharana |

Project leader for Iteration 3: Keertana Srinivasan.
