"""Iteration 3, Math Questions 1 and 2: the small worked example, step by step.

Five training messages, a five-word vocabulary and one new message. Every step of
the algorithm (Section 5 of the report) is computed with exact fractions
(fractions.Fraction) and printed, then checked a second way and against the
NumPy functions in methods.py, which also run the simulation and the real data.

Run from the iteration3 folder:   python src/small_example.py
"""
import sys
sys.dont_write_bytecode = True

from fractions import Fraction
import math
import numpy as np
from src.methods import (my_build_vocabulary, my_count_matrix,
                         my_nb_fit, my_nb_predict, my_posteriors, report_comparison)

CLASSES = ["ham", "spam"]


def show(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def fraction_row(values):
    return "  ".join(f"{str(v):>6s}" for v in values)


def main():
    # -----------------------------------------------------------------------
    show("Data: 5 training messages, 5-word vocabulary")
    messages = ["free win cash", "free free cash", "meet lunch", "lunch free", "meet win lunch"]
    y = np.array([1, 1, 0, 0, 0])          # 0 = ham, 1 = spam
    new_message = "free cash lunch"

    vocabulary = my_build_vocabulary(messages)
    X = my_count_matrix(messages, vocabulary)
    x_new = my_count_matrix([new_message], vocabulary)[0]
    V = len(vocabulary)

    for message, label in zip(messages, y):
        print(f"  {CLASSES[label]:5s} | {message}")
    print("Vocabulary:", vocabulary, " |V| =", V)
    print("Count matrix X (rows = messages, columns = words):")
    print("        " + "  ".join(f"{w:>6s}" for w in vocabulary))
    for d, row in enumerate(X):
        print(f"  d={d+1} " + "  ".join(f"{v:>6d}" for v in row) + f"   ({CLASSES[y[d]]})")
    print(f"New message: {new_message!r}   x_new = {x_new.tolist()}")

    # -----------------------------------------------------------------------
    show("Step 1: class counts and priors   pi_hat_c = n_c / n   (Proposition 2)")
    n = len(y)
    class_counts = [int(np.sum(y == c)) for c in (0, 1)]
    priors = [Fraction(class_counts[c], n) for c in (0, 1)]
    for c in (0, 1):
        print(f"  n_{CLASSES[c]} = {class_counts[c]},  pi_hat_{CLASSES[c]} = {class_counts[c]}/{n} = {priors[c]}")
    print("  Priors sum to", sum(priors))

    # -----------------------------------------------------------------------
    show("Step 2: word counts   N_cw = sum of x_dw over the class-c messages,  N_c = sum_w N_cw")
    word_counts = [[int(X[y == c, w].sum()) for w in range(V)] for c in (0, 1)]
    tokens = [sum(word_counts[c]) for c in (0, 1)]
    print("          " + "  ".join(f"{w:>6s}" for w in vocabulary) + "     N_c")
    for c in (0, 1):
        print(f"  {CLASSES[c]:6s}  " + "  ".join(f"{v:>6d}" for v in word_counts[c]) + f"   {tokens[c]:>5d}")

    # -----------------------------------------------------------------------
    show("Step 3a: unsmoothed MLE   theta_hat_cw = N_cw / N_c   (Proposition 2)")
    theta_hat = [[Fraction(word_counts[c][w], tokens[c]) for w in range(V)] for c in (0, 1)]
    for c in (0, 1):
        print(f"  {CLASSES[c]:6s}  " + fraction_row(theta_hat[c]))
    zero_pairs = [(CLASSES[c], vocabulary[w]) for c in (0, 1) for w in range(V) if word_counts[c][w] == 0]
    print("  Zero counts:", zero_pairs)
    unsmoothed = [priors[c] * math.prod(theta_hat[c][w] ** int(x_new[w]) for w in range(V)) for c in (0, 1)]
    print(f"  Without smoothing: score_ham = {unsmoothed[0]}, score_spam = {unsmoothed[1]}")
    print("  'cash' never occurs in ham and 'lunch' never occurs in spam, so both scores are 0")
    print("  and the posterior would be 0/0: the message cannot be classified (Proposition 3a).")

    # -----------------------------------------------------------------------
    show("Step 3b: Laplace smoothing   theta_tilde_cw = (N_cw + 1) / (N_c + |V|)   (Proposition 3)")
    alpha = 1
    theta = [[Fraction(word_counts[c][w] + alpha, tokens[c] + alpha * V) for w in range(V)] for c in (0, 1)]
    for c in (0, 1):
        print(f"  {CLASSES[c]:6s}  " + fraction_row(theta[c]) + f"    sum = {sum(theta[c])}")

    # -----------------------------------------------------------------------
    show("Steps 4-5 with probabilities   s_c = pi_c * prod_w theta_cw^x_w   (Proposition 1)")
    scores = []
    for c in (0, 1):
        factors = [theta[c][w] for w in range(V) for _ in range(int(x_new[w]))]
        score = priors[c] * math.prod(factors)
        scores.append(score)
        print(f"  s_{CLASSES[c]:4s} = {priors[c]} x " + " x ".join(str(f) for f in factors)
              + f" = {score}  ({float(score):.6f})")

    # -----------------------------------------------------------------------
    show("Steps 4-5 with logs   log s_c = log pi_c + sum_w x_w log theta_cw   (Proposition 4)")
    log_scores = []
    for c in (0, 1):
        terms = [math.log(priors[c])] + [int(x_new[w]) * math.log(theta[c][w]) for w in range(V) if x_new[w] > 0]
        log_scores.append(sum(terms))
        print(f"  log s_{CLASSES[c]:4s} = " + " + ".join(f"({t:.4f})" for t in terms) + f" = {sum(terms):.4f}")

    # -----------------------------------------------------------------------
    show("Steps 6-7: decision and posterior   Pr(c | x) = s_c / (s_ham + s_spam)")
    posterior = [scores[c] / (scores[0] + scores[1]) for c in (0, 1)]
    decision_prob = int(np.argmax([float(s) for s in scores]))
    decision_log = int(np.argmax(log_scores))
    for c in (0, 1):
        print(f"  Pr({CLASSES[c]} | x) = {posterior[c]}  ({float(posterior[c]):.4f})")
    print(f"  Decision with probabilities: {CLASSES[decision_prob]};  decision with logs: {CLASSES[decision_log]}")
    log_odds = log_scores[1] - log_scores[0]
    print(f"  Log posterior odds = {log_odds:.4f} > 0, so predict spam."
          f"  The posterior ({float(posterior[1]):.3f}) differs from the prior ({float(priors[1]):.3f}):"
          " the words changed the decision.")

    # -----------------------------------------------------------------------
    show("Check a second way")
    print("  1. Each smoothed row sums to 1:", [sum(theta[c]) == 1 for c in (0, 1)])
    print("  2. The posteriors sum to 1:", posterior[0] + posterior[1] == 1)
    print("  3. Lagrange condition (Proposition 2), N_cw / theta_hat_cw = N_c for every seen word:",
          [all(Fraction(word_counts[c][w]) / theta_hat[c][w] == tokens[c]
               for w in range(V) if word_counts[c][w] > 0) for c in (0, 1)])
    print("  4. Products and logs give the same decision:", decision_prob == decision_log)
    print("  5. log of the exact posterior odds equals the log-score difference:",
          math.isclose(math.log(posterior[1] / posterior[0]), log_odds, rel_tol=1e-12))

    # -----------------------------------------------------------------------
    show("Check against methods.py (the same functions run the simulation and the real data)")
    model = my_nb_fit(X, y, alpha=1.0)
    prediction, joint = my_nb_predict(model, x_new[None, :])
    model_posterior = my_posteriors(joint)
    report_comparison("priors vs exact fractions", model["priors"], [float(p) for p in priors])
    report_comparison("likelihoods vs exact fractions", model["likelihoods"],
                      [[float(v) for v in row] for row in theta])
    report_comparison("log-scores vs exact fractions", joint[0], log_scores)
    report_comparison("posteriors vs exact fractions", model_posterior[0], [float(p) for p in posterior])
    print("  Prediction from methods.py:", CLASSES[int(prediction[0])])

    # -----------------------------------------------------------------------
    show("Math Question 2: the underflow problem")
    probabilities = np.full(500, 0.01)
    running, first_zero = 1.0, None
    for i, value in enumerate(probabilities, start=1):
        running *= value                               # product in floating point
        if running == 0.0 and first_zero is None:
            first_zero = i
    log_score = float(np.sum(np.log(probabilities)))   # the same score as a sum of logs
    print(f"  Product of 500 probabilities of 0.01 = {running}   (true value 1e-1000)")
    print(f"  Smallest positive double = {np.nextafter(0, 1):.3e};  smallest normal double = {np.finfo(float).tiny:.3e}")
    print(f"  The running product first becomes exactly 0 after {first_zero} factors")
    print(f"  Sum of logs = 500 x log(0.01) = {log_score:.4f}   (finite, so classes can still be compared)")


if __name__ == "__main__":
    main()