# DS5020 Naive Bayes Spam Classifier

DS5020 Final Group Project: a Naive Bayes spam classifier built from counts. Every step is derived by hand, implemented from scratch with NumPy, and checked against scikit-learn.

**Team:** Debaswini Moharana, Keertana Srinivasan  
**Course:** DS5020 Introduction to Linear Algebra and Probability for Data Science, Fall 2026 (Dr. Fatema Nafa)  
**Dataset:** SMS Spam Collection (UCI Machine Learning Repository)

## Project overview

We classify SMS messages as **ham** or **spam** with a Multinomial Naive Bayes classifier. The project follows the course rule that the math leads: each method is derived in the Overleaf report, implemented in Python with NumPy only, and verified numerically on a tiny example solved with exact fractions, on simulated data (where the true answer is known) and on the real data.

Key results derived in the reports:

1. The Naive Bayes posterior from Bayes' theorem and conditional independence.
2. Maximum-likelihood estimates of the word probabilities (Lagrange multiplier).
3. Laplace smoothing and why zero counts break the classifier.
4. Logarithms and the linear decision rule.
5. Precision and recall as conditional probabilities.
6. The token-pair expectation used to test conditional independence on real data.

## Main results

| Check | Result |
|---|---|
| Small worked example vs exact fractions | differences ≤ 1.1 × 10⁻¹⁶; posterior 1728/3059 |
| Simulation: estimates vs true parameters | within 3 standard errors; error shrinks like 1/√n (log–log slope −0.504) |
| Simulation and real data vs scikit-learn `MultinomialNB` | 100% of predictions agree; log-probabilities differ only at rounding level |
| Real SMS test set (1,032 messages) | accuracy 97.97%, precision 95.73%, recall 87.50% |
| Assumption checks | conditional independence and equal message length do not hold exactly on real data |

## Repository structure

Each iteration has its own folder with its code, figures, report and project plan.

| Folder | Contents |
|---|---|
| [`iteration2/`](iteration2/) | Math foundation: notation, derivations, data loading and description, simulation check, report |
| [`iteration3/`](iteration3/) | Small worked example, from-scratch implementation (one function per algorithm step), verification against the truth and scikit-learn, assumption checks, report |

```
README.md
.gitignore
iteration2/
    README.md
    src/data_analysis.ipynb
    figures/
    docs/                 Iteration 2 report and project plan
iteration3/
    README.md
    src/                  methods.py, simulate.py, small_example.py,
                          check_simulation.py, run_real.py
    figures/
    docs/                 Iteration 3 report and project plan
```

Each iteration's `README.md` explains how to run its code.

## Quick start

1. Install the requirements (below).
2. Download the data (below) into the `data/` folder of the iteration you want to run.
3. Run the code:
   - **Iteration 2:** open `iteration2/src/data_analysis.ipynb`, restart the kernel and run all cells.
   - **Iteration 3:** from inside `iteration3/`, run
     ```
     python src/small_example.py
     python src/check_simulation.py
     python src/run_real.py
     ```

## Getting the data

Download the SMS Spam Collection from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Unzip it and place the file `SMSSpamCollection` in the `data/` folder of the iteration you want to run, for example `iteration3/data/SMSSpamCollection`. The file has 5,574 messages (4,827 ham, 747 spam), one per line, as `label<TAB>message`. The raw data is not stored in this repository.

## Requirements

Python 3 with `numpy`, `pandas`, `matplotlib`, `scipy` and `scikit-learn`:

```
pip install numpy pandas matplotlib scipy scikit-learn
```

The classifier itself uses NumPy only. SciPy and scikit-learn are used only to check our results.

## Team roles

| Member | Role |
|---|---|
| Debaswini Moharana | Math lead; project leader for Iteration 2 |
| Keertana Srinivasan | Data analysis and implementation lead; project leader for Iteration 3 |

Each member owns part of the math and part of the code, and the other member checks it. The owner and checker of every task are listed in the project plan in each iteration's `docs/` folder.
