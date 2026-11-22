"""Verify the from-scratch classifier on the Iteration 2 simulation.

Same model, parameter values and seed as Iteration 2. Two comparisons:
  * with the truth: differences should be about the size of the sampling error
    (within 3 standard errors) and shrink as n grows;
  * with a library (MultinomialNB, used only as a check): differences should be at
    rounding level, because both compute the same formula from the same data.
Then: error against n on log-log axes, assumption checks, and a stress test.
"""
import sys
sys.dont_write_bytecode = True

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.methods import (my_nb_fit, my_nb_predict, my_posteriors, my_confusion_matrix,
                         my_classification_metrics, my_token_pair_ratio, report_comparison)
from src.simulate import simulate_nb, simulation_train_test_split

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)


def main():
    X, y, theta_true, priors_true = simulate_nb(n_docs=2000, V=20, L=3, seed=0)
    X_train, X_test, y_train, y_test = simulation_train_test_split(X, y, train_size=1600)

    print("Messages:", len(y), " training:", len(y_train), " test:", len(y_test))
    print("Every message has exactly 3 tokens:", bool(np.all(X.sum(axis=1) == 3)))
    print("True class priors:", priors_true)
    print("True likelihood rows sum to 1:", theta_true.sum(axis=1))

    model = my_nb_fit(X_train, y_train, alpha=1.0)
    predictions, joint_scores = my_nb_predict(model, X_test)
    posteriors = my_posteriors(joint_scores)

    # Comparison 1: with the truth (sampling error)
    print("\nComparison with the truth:")
    report_comparison("priors vs truth", model["priors"], priors_true)
    report_comparison("P(w|c) vs truth", model["likelihoods"], theta_true)
    report_comparison("P(w|ham) vs truth", model["likelihoods"][0], theta_true[0])
    report_comparison("P(w|spam) vs truth", model["likelihoods"][1], theta_true[1])
    print("Word tokens per class N_c [ham, spam]:", model["tokens_per_class"])
    print("The spam row is less accurate: spam has about 20% of the messages, so it is estimated")
    print("from about a quarter as many tokens, and the standard error shrinks like 1/sqrt(N_c).")

    # Comparison 2: with the library (used only as a check)
    try:
        from sklearn.naive_bayes import MultinomialNB
    except ImportError as exc:
        raise ImportError("Install scikit-learn to run the required library check.") from exc

    library = MultinomialNB(alpha=1.0)
    library.fit(X_train, y_train)
    library_predictions = library.predict(X_test)
    library_joint = library.predict_joint_log_proba(X_test)

    print("\nPrediction agreement with MultinomialNB:",
          f"{np.mean(predictions == library_predictions):.4%}")
    joint_diff = report_comparison("Joint log scores vs MultinomialNB", joint_scores, library_joint)
    report_comparison("log priors vs MultinomialNB", model["log_priors"], library.class_log_prior_)
    report_comparison("log likelihoods vs MultinomialNB", model["log_likelihoods"], library.feature_log_prob_)
    report_comparison("posteriors vs MultinomialNB", posteriors, library.predict_proba(X_test))
    print("Both compute the same formulas from the same counts, so the differences are")
    print("floating-point rounding (machine precision is about 2.2e-16).")

    cm = my_confusion_matrix(y_test, predictions)
    print("\nConfusion matrix (rows actual ham/spam; columns predicted ham/spam):\n", cm)
    print("Metrics:", my_classification_metrics(cm))

    # Error against the truth for several sample sizes (log-log plot)
    print("\nParameter error versus sample size:")
    sizes = np.array([200, 500, 2000, 5000, 20000])
    errors = []
    for n in sizes:
        X_n, y_n, theta_n, _ = simulate_nb(n_docs=n, V=20, L=3, seed=0)
        n_train = int(0.8 * n)
        model_n = my_nb_fit(X_n[:n_train], y_n[:n_train], alpha=1.0)
        errors.append(np.linalg.norm(model_n["likelihoods"] - theta_n))
        print(f"n={n:6d}  likelihood L2 error={errors[-1]:.6f}")
    errors = np.array(errors)
    slope, intercept = np.polyfit(np.log(sizes), np.log(errors), 1)
    print(f"Fitted log-log slope = {slope:.3f}  (theory: -0.5)")

    plt.figure(figsize=(6, 4))
    plt.loglog(sizes, errors, "o-", label="observed error")
    plt.loglog(sizes, np.exp(intercept) * sizes ** -0.5, "--", label="slope -1/2 reference")
    plt.xlabel("number of simulated messages n")
    plt.ylabel("likelihood L2 error")
    plt.title(f"Error against the truth (fitted slope {slope:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "simulation_error_vs_n.png", dpi=200)
    plt.close()

    # Approximate three-standard-error diagnostic for n=2000.
    token_counts = model["word_counts"].sum(axis=1)
    se_theta = np.sqrt(theta_true * (1 - theta_true) / token_counts[:, None])
    z_theta = np.abs(model["likelihoods"] - theta_true) / se_theta
    print("\nFraction of word probabilities within 3 approximate SEs:",
          (z_theta <= 3).mean(axis=1))
    se_prior = np.sqrt(priors_true * (1 - priors_true) / len(y_train))
    print("Prior errors in SE units:",
          np.abs(model["priors"] - priors_true) / se_prior)

    # Assumption checks on the simulated data (they hold by construction)
    pair = max(
        (min(theta_true[0, a] * theta_true[0, b],
             theta_true[1, a] * theta_true[1, b]), a, b)
        for a in range(20) for b in range(a + 1, 20)
    )
    _, w1, w2 = pair
    print(f"\nA1 conditional independence, words {w1} and {w2} (token pairs within messages):")
    for c, name in ((0, "ham"), (1, "spam")):
        p_pair, product, ratio = my_token_pair_ratio(X, y, c, w1, w2)
        print(f"   {name}: Pr(w1,w2|c) = {p_pair:.5f}, Pr(w1|c)Pr(w2|c) = {product:.5f}, ratio = {ratio:.3f}")
    print("A2 bag of words: every position is drawn from the same theta_c by construction.")
    print("A3 independent messages: each message is drawn independently by construction.")
    print(f"A4 same distribution: spam share train = {y_train.mean():.4f}, test = {y_test.mean():.4f} (true 0.2)")
    lengths = X.sum(axis=1)
    print(f"A5 length: mean ham = {lengths[y == 0].mean():.2f}, spam = {lengths[y == 1].mean():.2f} (always 3)")
    zero = np.argwhere(model["word_counts"] == 0)
    print("A6 zero counts (class, word):", [(int(c), int(w)) for c, w in zero],
          "-> smoothed probabilities all positive:", bool(np.all(model["likelihoods"] > 0)))

    # Stress test: make class distributions progressively more similar.
    theta_average = (theta_true[0] + theta_true[1]) / 2
    print("\nStress test (class distributions become more similar):")
    rows = []
    for mixing in (0.0, 0.25, 0.50, 0.75, 1.0):
        theta_mixed = (1 - mixing) * theta_true + mixing * theta_average
        rng_mix = np.random.default_rng(1000)
        y_mix = rng_mix.choice(2, size=2000, p=priors_true)
        X_mix = np.array([rng_mix.multinomial(3, theta_mixed[c]) for c in y_mix])
        mix_model = my_nb_fit(X_mix[:1600], y_mix[:1600], alpha=1.0)
        mix_pred, _ = my_nb_predict(mix_model, X_mix[1600:])
        cm_mix = my_confusion_matrix(y_mix[1600:], mix_pred)
        metrics = my_classification_metrics(cm_mix)
        distance = np.linalg.norm(theta_mixed[0] - theta_mixed[1])
        print(f"mix={mixing:.2f} distance={distance:.6f} "
              f"accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
              f"recall={metrics['recall']:.4f}")
        rows.append((mixing, metrics["accuracy"], metrics["precision"], metrics["recall"]))
    print("When no message is predicted spam, precision = TP/(TP+FP) = 0/0 is undefined (nan).")

    rows = np.array(rows, dtype=float)
    plt.figure(figsize=(6, 4))
    plt.plot(rows[:, 0], rows[:, 2], "o-", label="precision")
    plt.plot(rows[:, 0], rows[:, 3], "s-", label="recall")
    plt.plot(rows[:, 0], rows[:, 1], "^--", label="accuracy")
    plt.xlabel("mixing level t (higher = more similar word distributions)")
    plt.ylabel("score")
    plt.ylim(0, 1.05)
    plt.title("Stress test: precision and recall as the classes become similar")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "stress_test_precision_recall.png", dpi=200)
    plt.close()

    # Rows for the verification table
    print("\nRows for the verification table")
    print(f"Simulation  P(w|c) vs truth: ||.||_2 = {np.linalg.norm(model['likelihoods'] - theta_true):.4f}")
    print(f"Simulation  predictions vs MultinomialNB: {np.mean(predictions == library_predictions):.2%}")
    print(f"Simulation  joint log-probs vs MultinomialNB: max|diff| = {joint_diff[0]:.1e}")
    print(f"Simulation  error vs n: slope = {slope:.3f}")


if __name__ == "__main__":
    main()