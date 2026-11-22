"""Simulation from the Iteration 2 two-class multinomial model."""
import sys
sys.dont_write_bytecode = True

import numpy as np

# y_d ~ Categorical(pi),  x_d ~ Multinomial(L, theta_{y_d}),  theta_c ~ Dirichlet(1)   (Definition 5)
def simulate_nb(n_docs=2000, V=20, L=3, seed=0):
    """Generate simulated messages and return X, y, true word probabilities, priors.

    The seed is applied once to the generator, matching the Iteration 2 notebook:
    first draw both Dirichlet word distributions, then class labels, then messages.
    """
    rng = np.random.default_rng(seed)
    theta_true = np.vstack([
        rng.dirichlet(np.ones(V)),  # ham
        rng.dirichlet(np.ones(V)),  # spam
    ])
    priors_true = np.array([0.8, 0.2])
    y = rng.choice(2, size=n_docs, p=priors_true)
    X = np.array([rng.multinomial(L, theta_true[c]) for c in y])
    return X, y, theta_true, priors_true


def simulation_train_test_split(X, y, train_size=1600):
    """Use the fixed first-80% / last-20% split specified in the handout (1600 / 400 for n = 2000)."""
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]
