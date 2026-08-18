"""
stats_utils.py — the small amount of statistics this study needs, written out
rather than imported, so every number in the writeup can be audited.

Sample sizes here are 3-6 runs per condition. That rules out anything
asymptotic: the tests below are exact (enumerate every assignment) or
resampling-based, and the effect sizes are non-parametric.
"""

import itertools
import math

import numpy as np


def mannwhitney_u(a, b):
    """Exact two-sided Mann-Whitney U test.

    Returns (U, p). U is computed for sample `a`. The null distribution is
    enumerated over all C(n+m, n) label assignments, which is exact and cheap
    for the sizes used here (C(12,6) = 924).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    pooled = np.concatenate([a, b])

    def u_stat(idx_a):
        x = pooled[list(idx_a)]
        y = pooled[[i for i in range(n + m) if i not in set(idx_a)]]
        # count wins with ties at half
        u = 0.0
        for xi in x:
            u += (xi > y).sum() + 0.5 * (xi == y).sum()
        return u

    observed = u_stat(tuple(range(n)))
    centre = n * m / 2.0
    extreme = 0
    total = 0
    for idx in itertools.combinations(range(n + m), n):
        total += 1
        if abs(u_stat(idx) - centre) >= abs(observed - centre) - 1e-12:
            extreme += 1
    return observed, extreme / total


def cliffs_delta(a, b):
    """Non-parametric effect size in [-1, 1]: P(a > b) - P(a < b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def bootstrap_ci(x, stat=np.mean, n_boot=20000, alpha=0.05, seed=0):
    """Percentile bootstrap CI. With n=6 this is wide on purpose."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    draws = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(draws, 100 * alpha / 2)), \
        float(np.percentile(draws, 100 * (1 - alpha / 2)))


def sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def describe(x):
    x = np.asarray(x, dtype=float)
    lo, hi = bootstrap_ci(x) if len(x) > 1 else (float("nan"), float("nan"))
    return {"n": int(len(x)), "mean": float(x.mean()), "sem": sem(x),
            "median": float(np.median(x)), "min": float(x.min()),
            "max": float(x.max()), "ci_lo": lo, "ci_hi": hi}


def compare(a, b):
    """Everything reported for a two-condition comparison."""
    u, p = mannwhitney_u(a, b)
    return {"a": describe(a), "b": describe(b),
            "diff_of_means": float(np.mean(a) - np.mean(b)),
            "mannwhitney_u": float(u), "p_two_sided": float(p),
            "cliffs_delta": float(cliffs_delta(a, b))}


def _rank(x):
    x = np.asarray(x, dtype=float)
    order = x.argsort()
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def spearman(x, y):
    """Spearman rank correlation, ties averaged."""
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / d) if d > 0 else float("nan")
