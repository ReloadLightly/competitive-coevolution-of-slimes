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


# --------------------------------------------------------------------------
# Added for the confirmatory replication (12 runs per arm). Enumerating
# C(24, 12) = 2.7 million assignments in Python is too slow, so the same exact
# null distribution is counted by dynamic programming instead.
# --------------------------------------------------------------------------
def mannwhitney_dp(a, b, alternative="two-sided"):
    """Exact Mann-Whitney U test, same definition as mannwhitney_u.

    alternative="two-sided" (the default, as mannwhitney_u) or "greater"
    (a tends to exceed b: P(U >= U_observed) under the null).

    Under the null every n-subset of the pooled mid-ranks is equally likely to
    be sample a. U = (rank sum of a) - n(n+1)/2 with ties at half, so counting
    n-subsets by their (doubled, hence integer) rank sum gives the exact null
    distribution of U, ties included. Returns (U, p).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    pooled = np.concatenate([a, b])
    r2 = np.rint(2 * (_rank(pooled) + 1)).astype(int)    # doubled mid-ranks
    top = int(r2.sum())
    # ways[k][s]: number of k-subsets with doubled rank sum s (exact integers)
    ways = [[0] * (top + 1) for _ in range(n + 1)]
    ways[0][0] = 1
    for r in r2:
        for k in range(min(n, len(r2)) - 1, -1, -1):
            row, nxt = ways[k], ways[k + 1]
            for s in range(top - r, -1, -1):
                if row[s]:
                    nxt[s + r] += row[s]
    obs2 = int(r2[:n].sum())
    centre2 = n * (n + m + 1)            # doubled expected rank sum of a
    if alternative == "greater":
        extreme = sum(c for s, c in enumerate(ways[n]) if c and s >= obs2)
    else:
        extreme = sum(c for s, c in enumerate(ways[n])
                      if c and abs(s - centre2) >= abs(obs2 - centre2))
    u = obs2 / 2.0 - n * (n + 1) / 2.0
    return u, extreme / math.comb(n + m, n)


def sign_test_greater(x):
    """Exact one-sided sign test that the median of x is above zero.

    Zeros are dropped. Returns (positives, informative, p)."""
    x = np.asarray(x, dtype=float)
    k = int((x > 0).sum())
    n = int((x != 0).sum())
    if n == 0:
        return 0, 0, 1.0
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return k, n, float(p)


def fisher_less(k1, n1, k2, n2):
    """Exact one-sided Fisher test that rate 1 (k1/n1) is below rate 2 (k2/n2).

    P(X <= k1) under the hypergeometric null with the margins fixed."""
    K, N = k1 + k2, n1 + n2
    total = math.comb(N, n1)
    p = sum(math.comb(K, x) * math.comb(N - K, n1 - x)
            for x in range(max(0, n1 - (N - K)), k1 + 1)) / total
    return float(p)


def signflip_greater(d, n_mc=100_000, seed=0):
    """One-sided paired sign-flip permutation test that mean(d) > 0.

    Under the null each difference is equally likely to have either sign.
    Exact (all 2^n sign patterns) for n <= 20; otherwise Monte Carlo with a
    fixed seed, p = (hits + 1) / (n_mc + 1). Returns (mean, p)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    obs = float(d.mean())
    if n <= 20:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        means = (signs * np.abs(d)).mean(axis=1)
        return obs, float((means >= obs - 1e-12).mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_mc // 10_000):
        signs = rng.choice((-1.0, 1.0), size=(10_000, n))
        hits += int(((signs * np.abs(d)).mean(axis=1) >= obs - 1e-12).sum())
    return obs, (hits + 1) / (n_mc + 1)


def holm(pvals, alpha=0.05):
    """Holm step-down: which hypotheses are rejected at family-wise alpha."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    reject = [False] * len(pvals)
    for step, i in enumerate(order):
        if pvals[i] <= alpha / (len(pvals) - step):
            reject[i] = True
        else:
            break
    return reject
