"""
GUARD core — Group Uncertainty And Residual Decomposition.

Two-axis group-level uncertainty metric for label noise (research plan §2).
  Axis A (homogeneity, label-free) : H(pbar) = W(g) + D(g)   [BALD / Jensen-gap identity]
  Axis B (contamination, label-aware):
        C(g)     = JS(q_g || pbar)           (claim-belief divergence, magnitude)
        kappa(g) = 1 - H(rtilde)/log K       (residual direction concentration)

All entropies use natural log; JS uses base-2 (range [0, 1]).
This module is import-safe and has no heavy dependencies (numpy only).
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------- primitives
def entropy(p, axis=-1):
    """Shannon entropy (natural log)."""
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def js_divergence(q, p):
    """Jensen-Shannon divergence, base-2 (range [0, 1]). q, p are 1-D pmfs."""
    q = np.clip(np.asarray(q, dtype=np.float64), EPS, 1.0)
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0)
    m = 0.5 * (q + p)
    return float(0.5 * np.sum(q * np.log2(q / m)) + 0.5 * np.sum(p * np.log2(p / m)))


# ---------------------------------------------------------------- axes
def axis_A(P):
    """Homogeneity decomposition. P: (n, K) predicted distributions.
    Returns (W, D, H_pbar, pbar) with H_pbar = W + D exactly, D >= 0."""
    P = np.asarray(P, dtype=np.float64)
    pbar = P.mean(axis=0)
    W = float(entropy(P, axis=1).mean())   # within-member ambiguity (aleatoric)
    H_pbar = float(entropy(pbar))          # total marginal uncertainty
    D = H_pbar - W                         # between-member disagreement (epistemic) >= 0
    return W, D, H_pbar, pbar


def axis_B(P, a, K):
    """Contamination decomposition.
    P: (n, K) predictions ; a: (n,) assigned (noisy) integer labels in [0, K).
    Returns (C, kappa, q, pbar)."""
    P = np.asarray(P, dtype=np.float64)
    a = np.asarray(a)
    n = len(a)
    q = np.bincount(a, minlength=K).astype(np.float64) / n   # claim (assigned-label) pmf
    pbar = P.mean(axis=0)                                    # belief pmf
    C = js_divergence(q, pbar)
    r = np.clip(pbar - q, 0.0, None)                         # belief-exceeds-claim residual
    if r.sum() <= EPS:
        kappa = 0.0
    else:
        rt = r / r.sum()
        kappa = float(1.0 - entropy(rt) / np.log(K))
    return C, kappa, q, pbar


def guard_tuple(P, a, K, lam=(1.0, 1.0, 0.0, 0.0)):
    """Full GUARD 4-tuple plus the combined detection score.
    lam = (lambda_C, lambda_Ckappa, lambda_W, lambda_D). Defaults to the core
    detection terms C + C*kappa (weights are an ablation choice, plan §2.7)."""
    W, D, H_pbar, _ = axis_A(P)
    C, kappa, _, _ = axis_B(P, a, K)
    lC, lCk, lW, lD = lam
    score = lC * C + lCk * C * kappa + lW * W + lD * D
    return dict(W=W, D=D, H_pbar=H_pbar, C=C, kappa=kappa, score=score)


# ---------------------------------------------------------------- group driver
def guard_by_group(P, a, group_ids, K, lam=(1.0, 1.0, 0.0, 0.0), min_size=2):
    """Compute the GUARD tuple for every group.

    P         : (N, K) array-like (dense np.ndarray or np.memmap) of predictions.
    a         : (N,) assigned integer labels.
    group_ids : (N,) group identifier per item (any hashable dtype).
    Returns a list of dict rows: group, n, W, D, H_pbar, C, kappa, score.
    """
    a = np.asarray(a)
    group_ids = np.asarray(group_ids)
    rows = []
    for gid in np.unique(group_ids):
        idx = np.where(group_ids == gid)[0]
        if len(idx) < min_size:
            continue
        Pg = np.asarray(P[idx], dtype=np.float64)
        t = guard_tuple(Pg, a[idx], K, lam=lam)
        t.update(group=gid, n=int(len(idx)))
        rows.append(t)
    return rows


def individual_scores(P, a):
    """Per-item claim-belief mismatch score (baseline for aggregation)."""
    P = np.asarray(P, dtype=np.float64)
    a = np.asarray(a)
    n = len(a)
    return 1.0 - P[np.arange(n), a]


# ---------------------------------------------------------------- routing
def review_budget_curve(scores, is_error, budgets=None):
    """Recall of true errors as a function of the top-k review budget.
    scores  : higher => more suspicious. is_error : bool ground truth.
    Returns (budgets, recall) arrays."""
    scores = np.asarray(scores)
    is_error = np.asarray(is_error).astype(bool)
    order = np.argsort(-scores)
    err_sorted = is_error[order]
    total = max(err_sorted.sum(), 1)
    cum = np.cumsum(err_sorted)
    n = len(scores)
    if budgets is None:
        budgets = np.linspace(0.01, 1.0, 100)
    recall = [cum[min(int(round(b * n)), n) - 1] / total for b in budgets]
    return np.asarray(budgets), np.asarray(recall)
