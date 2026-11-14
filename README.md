# DS5020 Naive Bayes Spam Classifier

DS5020 Final Group Project: a Naive Bayes spam classifier built from counts. Every step is derived by hand, implemented from scratch with NumPy, and checked against scikit-learn.

**Team:** Debaswini Moharana, Keertana Srinivasan  
**Course:** DS5020 Introduction to Linear Algebra and Probability for Data Science, Fall 2026 (Dr. Fatema Nafa)  
**Dataset:** SMS Spam Collection (UCI Machine Learning Repository)

## Project overview

We classify SMS messages as **ham** or **spam** with a Multinomial Naive Bayes classifier. The project follows the course rule that the math leads: each method is derived in the Overleaf report, implemented in Python with NumPy only, and verified numerically on simulated data (where the true answer is known) and on the real data.

Key results derived in the report:

1. The Naive Bayes posterior from Bayes' theorem and conditional independence.
2. Maximum-likelihood estimates of the word probabilities (Lagrange multiplier).
3. Laplace smoothing and why zero counts break the classifier.
4. Logarithms and the linear decision rule.
5. Precision and recall as conditional probabilities.

## Repository structure

Each iteration has its own folder.

| Folder | Contents |
|---|---|
| [`iteration2/`](iteration2/) | Math foundation: data loading and description, simulation check, report |
| `iteration3/` | Small worked example, from-scratch implementation and verification (coming next) |

## Team roles

| Member | Role |
|---|---|
| Debaswini Moharana | Math lead; project leader for Iteration 2 |
| Keertana Srinivasan | Data analysis and implementation lead; project leader for Iteration 3 |

Each derivation and analysis has an owner and a checker; see the project plan in each iteration's `docs/` folder.

## Getting the data

Download the SMS Spam Collection from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Unzip it and place the file `SMSSpamCollection` in the `data/` folder of the iteration you want to run, for example `iteration2/data/SMSSpamCollection`. The file has 5,574 messages (4,827 ham, 747 spam), one per line, as `label<TAB>message`.

## Requirements

Python 3 with `numpy`, `pandas`, `matplotlib` and `scipy`:

```
pip install numpy pandas matplotlib scipy
```
