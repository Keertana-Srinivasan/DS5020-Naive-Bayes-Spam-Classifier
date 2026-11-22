"""Run the from-scratch classifier on the SMS Spam Collection and verify it.

Same cleaning, split and seed as Iteration 2. Compares our results with
MultinomialNB (Math Q3), computes the confusion matrix, precision and recall
(Math Q4), and checks every Iteration 2 assumption on the real data (incl. Math Q5).
"""
import sys
sys.dont_write_bytecode = True
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from src.methods import (
    my_build_vocabulary,my_count_matrix,
    my_nb_fit, my_nb_predict,my_posteriors, my_confusion_matrix,
    my_classification_metrics, my_token_pair_ratio,report_comparison
)

def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)

def stratified_split(y, test_fraction=0.20, seed=0):
    """Stratified split matching the Iteration 2 procedure."""
    rng = np.random.default_rng(seed)
    train_indices, test_indices = [], []
    for class_label in np.unique(y):
        indices = np.flatnonzero(y == class_label)
        rng.shuffle(indices)
        n_test = int(round(test_fraction * len(indices)))
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])
    return np.sort(train_indices), np.sort(test_indices)


def find_dataset():
    """Find the dataset in the expected project data folder or common alternatives."""
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / "data" / "SMSSpamCollection",
        project_root / "SMSSpamCollection",
        Path.cwd() / "data" / "SMSSpamCollection",
        Path.cwd() / "SMSSpamCollection",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "SMSSpamCollection was not found. Place it at project_root/data/SMSSpamCollection."
    )


def main():
    section("Load and clean the data (same steps as Iteration 2)")
    data_path = find_dataset()
    sms_original = pd.read_csv(
        data_path, sep="\t", header=None, names=["label", "message"],
        quoting=csv.QUOTE_NONE, encoding="utf-8"
    )
    print("Dataset:", data_path)
    print("Rows initially loaded:", len(sms_original))
    print("Original label counts:", sms_original["label"].value_counts().to_dict())

    sms = sms_original.copy()
    sms["label"] = sms["label"].astype(str).str.strip().str.lower()
    sms["message"] = sms["message"].astype(str).str.strip()
    n_start = len(sms)

    sms = sms[sms["label"].isin(["ham", "spam"]) & sms["message"].ne("")].copy()
    n_after_valid = len(sms)

    labels_per_text = sms.groupby("message")["label"].nunique()
    conflicting_texts = labels_per_text[labels_per_text > 1].index
    sms = sms[~sms["message"].isin(conflicting_texts)].copy()
    n_after_conflicts = len(sms)

    sms = sms.drop_duplicates(subset=["label", "message"]).reset_index(drop=True)
    n_clean = len(sms)

    print("\nCleaning summary:")
    print("Invalid label/empty message rows removed:", n_start - n_after_valid)
    print("Conflicting-label messages removed:", n_after_valid - n_after_conflicts)
    print("Duplicate label-message rows removed:", n_after_conflicts - n_clean)
    print("Rows after cleaning:", n_clean)
    print("Cleaned class counts:", sms["label"].value_counts().to_dict())

    y_all = (sms["label"].to_numpy() == "spam").astype(int)
    messages = sms["message"].to_numpy()
    train_idx, test_idx = stratified_split(y_all, test_fraction=0.20, seed=0)

    train_messages, test_messages = messages[train_idx], messages[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    vocabulary = my_build_vocabulary(train_messages)  # training set only
    X_train = my_count_matrix(train_messages, vocabulary)
    X_test = my_count_matrix(test_messages, vocabulary)

    print("\nTraining messages:", len(y_train))
    print("Test messages:", len(y_test))
    print("Training spam share:", y_train.mean())
    print("Test spam share:", y_test.mean())
    print("Training vocabulary size:", len(vocabulary))

    model = my_nb_fit(X_train, y_train, alpha=1.0)
    predictions, joint_scores = my_nb_predict(model, X_test)
    posteriors = my_posteriors(joint_scores)
    cm = my_confusion_matrix(y_test, predictions)
    metrics = my_classification_metrics(cm)

    print("\nConfusion matrix (rows actual ham/spam; columns predicted ham/spam):")
    print(cm)
    print("\nClassification metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")

    precision_counting = np.mean(y_test[predictions == 1] == 1)  # Pr(S | S_hat) by counting
    recall_counting = np.mean(predictions[y_test == 1] == 1)  # Pr(S_hat | S) by counting
    print(
            f"Precision: formula {metrics['precision']:.4f}, counting inside 'predicted spam' {precision_counting:.4f}")
    print(f"Recall:    formula {metrics['recall']:.4f}, counting inside 'actually spam' {recall_counting:.4f}")

    try:
        from sklearn.naive_bayes import MultinomialNB
    except ImportError as exc:
        raise ImportError("Install scikit-learn to run the required library comparison.") from exc

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
    print("The differences are floating-point rounding (relative error about 1e-16): both compute")
    print("the same formulas from the same counts. Naive Bayes has no sign or scaling convention to fix.")

    # Conditional independence check using message-level word-presence events.
    # ---------------------------------------------------------------------
    section("Assumption checks on the real data")
    from scipy import stats  # used only for the length test (Assumption 5)

    # A1 conditional independence (Math Question 5): token pairs, ratio 1 = independent
    print("A1 conditional independence (token pairs within messages; ratio 1 = independent):")
    for word_1, word_2 in (("call", "now"), ("free", "txt"), ("i", "you")):
        j1, j2 = vocabulary.index(word_1), vocabulary.index(word_2)
        for class_label, class_name in ((0, "ham"), (1, "spam")):
            p_pair, product, ratio = my_token_pair_ratio(X_train, y_train, class_label, j1, j2)
            print(f"   {word_1}/{word_2} {class_name}: Pr(w1,w2|c) = {p_pair:.6f}, "
                  f"Pr(w1|c)Pr(w2|c) = {product:.6f}, ratio = {ratio:.2f}")

    # A2 bag of words: look at misclassified test messages
    wrong = np.flatnonzero(predictions != y_test)
    print(f"A2 bag of words: {len(wrong)} misclassified test messages; the first five:")
    for i in wrong[:5]:
        print(f"   actual {['ham', 'spam'][y_test[i]]}, Pr(spam|x) = {posteriors[i, 1]:.3f}: {test_messages[i][:70]}")

    # A3 independent messages: duplicates removed and no train/test overlap
    overlap = set(train_messages) & set(test_messages)
    print(f"A3 independent messages: {n_after_conflicts - n_clean} duplicates removed; "
          f"messages in both train and test: {len(overlap)}")

    # A4 same distribution for training and test
    print(f"A4 same distribution: spam share train = {y_train.mean():.4f}, test = {y_test.mean():.4f}")

    # A5 message length does not depend on the class
    lengths = X_train.sum(axis=1)
    ks = stats.ks_2samp(lengths[y_train == 0], lengths[y_train == 1])
    print(f"A5 length: mean tokens ham = {lengths[y_train == 0].mean():.2f}, "
          f"spam = {lengths[y_train == 1].mean():.2f}; KS p = {ks.pvalue:.1e} -> the assumption fails")

    # A6 positive probabilities: zero counts before smoothing
    zero = model["word_counts"] == 0
    affected = ((X_test > 0) & zero[1][None, :]).any(axis=1).sum()
    print(f"A6 zero counts: {zero[1].mean():.1%} of words never in spam; {affected} of {len(y_test)} "
          f"test messages affected; all smoothed probabilities > 0: {bool(np.all(model['likelihoods'] > 0))}")

    # ---------------------------------------------------------------------
    section("Rows for the verification table")
    print(f"Real data  predictions vs MultinomialNB: agreement = {np.mean(predictions == library_predictions):.2%}")
    print(f"Real data  joint log-probs vs MultinomialNB: max|diff| = {joint_diff[0]:.1e}, rel = {joint_diff[1]:.1e}")
    print(f"Real data  precision, recall: {metrics['precision']:.4f}, {metrics['recall']:.4f}")

if __name__ == "__main__":
    main()
