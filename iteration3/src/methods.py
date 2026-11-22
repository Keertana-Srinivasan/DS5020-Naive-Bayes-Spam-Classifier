"""Naive Bayes classifier functions implemented from scratch with NumPy.

Class labels: 0 = ham, 1 = spam.
The classifier uses a multinomial word-count model with Laplace smoothing.
"""
import sys
sys.dont_write_bytecode = True
import re
import numpy as np





#tokens of a message = runs of letters and digits after lower-casing
def my_tokenize(message):
    return re.findall(r"[a-z0-9]+", str(message).lower())


# V = sorted set of all words in the training messages (training data only)
def my_build_vocabulary(messages):
    return sorted({word for message in messages for word in my_tokenize(message)})

#words outside the vocabulary are ignored

def my_count_matrix(messages,vocabulary):
    word_to_index = {word: j for j , word in enumerate(vocabulary)}
    X = np.zeros((len(messages), len(vocabulary)),  dtype=np.int64)
    for d , message in enumerate(messages):
        for word in my_tokenize(message):
            j = word_to_index.get(word)
            if j is not None:
                X[d, j] += 1
    return X

#Training

#step1: n_c = sum_d 1(y_d = c),   pi_hat_c = n_c / n   (Proposition 2)
def my_class_priors(y,n_classes=2):
    one_hot = np.eye(n_classes,dtype= np.int64)[y]
    class_counts = one_hot.sum(axis=0)
    return class_counts, class_counts /len(y)

#step 2:  N = Y^T X,   N_cw = sum_{d: y_d=c} x_dw,   N_c = sum_w N_cw
def my_word_counts(X,y,n_classes=2):
    one_hot = np.eye(n_classes,dtype= np.int64)[y]
    word_counts = one_hot.T @ X
    return word_counts, word_counts.sum(axis=1)

#step 3: theta_tilde_cw = (N_cw + alpha) / (N_c + alpha |V|)   (Proposition 3)
def my_smoothed_likelihood(word_counts,alpha=1.0):
    vocab_size = word_counts.shape[1]
    tokens_per_class = word_counts.sum(axis=1)
    return (word_counts + alpha) / (tokens_per_class[:,None] + alpha * vocab_size)

#step 4: b_c = log pi_hat_c,   A_cw = log theta_tilde_cw
def my_log_parameters(prior,likelihood):
    return np.log(prior), np.log(likelihood)



# Steps 1-4: check the inputs, then call the four step functions
def my_nb_fit(X, y, alpha=1.0, n_classes=2):
    """Fit Multinomial Naive Bayes.

    Equations:
        n_c = sum_d 1(y_d = c)
        pi_c = n_c / n
        N_cw = sum_d 1(y_d = c) X_dw
        theta_cw = (N_cw + alpha) / (N_c + alpha * V)
    """
    X = np.asarray(X, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)

    if X.ndim != 2:
        raise ValueError("X must be a 2D document-word count matrix.")
    if len(y) != X.shape[0]:
        raise ValueError("X and y must contain the same number of documents.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if np.any(X < 0):
        raise ValueError("Word counts cannot be negative.")
    if np.any((y < 0) | (y >= n_classes)):
        raise ValueError("Class labels are outside the allowed range.")

    class_counts, priors = my_class_priors(y, n_classes)  # step 1
    if np.any(class_counts == 0):
        raise ValueError("Every class must occur in the training data.")
    word_counts, tokens_per_class = my_word_counts(X, y, n_classes)  # step 2
    likelihoods = my_smoothed_likelihood(word_counts, alpha)  # step 3
    log_priors, log_likelihoods = my_log_parameters(priors, likelihoods)  # step 4

    return {
        "class_counts": class_counts,
        "word_counts": word_counts,
        "tokens_per_class": tokens_per_class,
        "priors": priors,
        "likelihoods": likelihoods,
        "log_priors": log_priors,
        "log_likelihoods": log_likelihoods,
    }

# Step 5:  S = X A^T + 1 b^T,   S_dc = log pi_c + sum_w x_dw log theta_cw   (Proposition 4)
def my_nb_joint_log_scores(model, X):
    """Compute log P(c) + sum_w X_dw log P(w|c) for every document/class."""
    X = np.asarray(X, dtype=np.int64)
    return X @ model["log_likelihoods"].T + model["log_priors"]


# Step 6:  y_hat_d = argmax_c S_dc
def my_nb_predict(model, X):
    """Predict the class with the largest joint log score."""
    scores = my_nb_joint_log_scores(model, X)
    return np.argmax(scores, axis=1), scores

# Step 7:  Pr(c | x_d) = exp(S_dc - M_d) / sum_c' exp(S_dc' - M_d),  M_d = max_c S_dc   (Proposition 1)
def my_posteriors(scores):
    shifted = np.exp(scores - scores.max(axis=1,keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)

# Step 8a: cm = [[TN, FP], [FN, TP]], spam (1) = positive class
def my_confusion_matrix(y_true, y_pred):
    """Return a 2x2 confusion matrix; rows=actual, columns=predicted."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        cm[actual, predicted] += 1
    return cm

# Step 8b: precision = TP/(TP+FP) = Pr(S | S_hat),  recall = TP/(TP+FN) = Pr(S_hat | S)   (Proposition 5)
def my_classification_metrics(cm):
    """Calculate accuracy, precision and recall, treating spam (1) as positive."""
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    total = cm.sum()
    return {
        "accuracy": (tp + tn) / total if total else np.nan,
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }



# ---------------------------------------------------------------------------
# Assumption check (Math Question 5): conditional independence of tokens
# ---------------------------------------------------------------------------

# For class c and two different words a, b, over all pairs of different positions i != j:
#   Pr_hat(W_i = a, W_j = b | c) = sum_d x_da x_db / sum_d L_d (L_d - 1)
#   Pr_hat(W = a | c)            = sum_d x_da / sum_d L_d
#   ratio = Pr_hat(a, b | c) / (Pr_hat(a | c) Pr_hat(b | c))
# Under conditional independence (Assumption 1) E[x_da x_db | L_d] = L_d (L_d - 1) theta_ca theta_cb,
# so the ratio is close to 1. A ratio far from 1 shows that the two words are dependent.
def my_token_pair_ratio(X, y, c, a, b):
    X_c = np.asarray(X, dtype=np.int64)[np.asarray(y) == c]
    lengths = X_c.sum(axis=1)
    p_pair = (X_c[:, a] * X_c[:, b]).sum() / (lengths * (lengths - 1)).sum()
    p_a = X_c[:, a].sum() / lengths.sum()
    p_b = X_c[:, b].sum() / lengths.sum()
    return p_pair, p_a * p_b, p_pair / (p_a * p_b)



# max |mine - reference|  and  relative error ||mine - reference|| / ||reference||
def report_comparison(name, mine, reference):
    """Print maximum absolute difference and relative L2 error."""
    mine = np.asarray(mine, dtype=float)
    reference = np.asarray(reference, dtype=float)
    max_abs = np.max(np.abs(mine - reference))
    denom = np.linalg.norm(reference)
    rel = np.linalg.norm(mine - reference) / denom if denom else np.linalg.norm(mine-reference)
    print(f"{name:38s} max|diff| = {max_abs:.3e}   rel = {rel:.3e}")
    return max_abs, rel
