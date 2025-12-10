import numpy as np
from typing import Tuple
from scipy import stats


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def ks_test(
    samples1: np.ndarray,
    samples2: np.ndarray,
) -> Tuple[float, float]:
    """
    Kolmogorov–Smirnov test on 1D samples (e.g., probs for positive class).
    Returns (statistic, p-value).
    """
    stat, pval = stats.ks_2samp(samples1, samples2)
    return stat, pval


def welch_ttest(
    samples1: np.ndarray,
    samples2: np.ndarray,
) -> Tuple[float, float]:
    """
    Welch’s t-test for unequal variances.
    """
    stat, pval = stats.ttest_ind(samples1, samples2, equal_var=False)
    return stat, pval


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    TV(P, Q) = 0.5 * sum_x |P(x) - Q(x)|
    Here we assume p, q are probability vectors over classes.
    """
    return 0.5 * np.abs(p - q).sum(axis=1).mean()


def empirical_epsilon(p: np.ndarray, q: np.ndarray, eps_floor: float = 1e-12) -> float:
    """
    Very rough empirical epsilon:
    epsilon_hat = max_x log(P(x) / Q(x)), approximated via
    per-class probabilities with floor to avoid division by zero.

    This is NOT the Ding et al. epsilon, only a heuristic upper bound.
    """
    p_clipped = np.clip(p, eps_floor, 1.0)
    q_clipped = np.clip(q, eps_floor, 1.0)
    ratio = p_clipped / q_clipped
    return float(np.log(ratio).max())


def accuracy_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute classification accuracy from probability distributions.
    probs: array [N, K]
    labels: array [N]
    """
    preds = probs.argmax(axis=1)
    return float((preds == labels).mean())

