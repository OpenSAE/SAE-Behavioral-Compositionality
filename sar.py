"""SAE Additivity Residuals (SAR).

Keep or remove K selected SAE contributions, read one behavior g(S) for each coalition S, fit the best
first order and pairwise models on some coalitions, and measure what they miss on the others.
SAR_r is the held-out fraction of behavioral variance left unexplained at interaction order r.

Numerical conventions follow the code behind the paper and are kept fixed so results stay comparable:
  * SAR_r = L_r / (V + eps) with eps = 1e-8. SAR is scale free only while V is much larger than eps,
    which holds for logit scale behaviors. A warning is raised when eps exceeds 1% of V.
  * rho_2 is NaN when SAR_1 <= 1e-4, where the pairwise share is not meaningful.
  * Sign error compares the three valued signs of e(S) = g(S) - g(N) and e_hat(S) = a(S) - a(N) with no
    tolerance. Both zero is not an error, exactly one zero is an error. Otherwise this equals e * e_hat < 0.
"""
import warnings
from itertools import combinations

import numpy as np


def coalitions(K=16, M=384, p=0.5, n_fit=256, seed=0):
    """M random coalitions (each feature kept with probability p) and a fit / held-out split."""
    if not (K >= 1 and 0 < n_fit < M and 0 < p < 1):
        raise ValueError("need K >= 1, 0 < n_fit < M and 0 < p < 1")
    rng = np.random.default_rng(seed)
    masks = rng.random((M, K)) < p
    order = rng.permutation(M)
    return masks, order[:n_fit], order[n_fit:]


def rotation(K, seed=0):
    """Random orthogonal Q with Q @ 1 = 1: rotates the K contributions but keeps their span and their sum."""
    if K < 2:
        raise ValueError("a rotation needs K >= 2")
    e = np.ones(K) / np.sqrt(K)
    U = np.column_stack([e, np.linalg.qr(np.eye(K) - np.outer(e, e))[0][:, : K - 1]])  # [1/sqrt(K), its complement]
    R, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((K - 1, K - 1)))
    Q = np.eye(K)
    Q[1:, 1:] = R * np.sign(np.diag(r))                 # Haar rotation of the complement
    return U @ Q @ U.T


def design(masks, order):
    """Columns 1{B in S} for |B| <= order: intercept, single features, then pairs. order is 1 or 2."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    s = np.asarray(masks, dtype=np.float64)
    cols = [np.ones((len(s), 1)), s]
    if order == 2:
        cols += [s[:, [i]] * s[:, [j]] for i, j in combinations(range(s.shape[1]), 2)]
    return np.hstack(cols)


def _spearman(a, b):
    def rank(x):                                        # average ranks over ties
        r = np.argsort(np.argsort(x, kind="stable")).astype(np.float64)
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        return (np.bincount(inv, weights=r) / cnt)[inv]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def _check(g, masks, fit, held, g_full):
    if g.ndim != 1 or masks.ndim != 2 or len(g) != len(masks):
        raise ValueError(f"g must have shape (M,) and masks (M, K); got {g.shape} and {masks.shape}")
    if not np.all(np.isfinite(g)) or (g_full is not None and not np.isfinite(g_full)):
        raise ValueError("behavior values must be finite")
    fit, held = np.asarray(fit), np.asarray(held)
    for name, idx in (("fit", fit), ("held", held)):
        if idx.ndim != 1 or len(idx) < 2 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f"{name} must be a 1-D integer index array with at least 2 entries")
        if idx.min() < 0 or idx.max() >= len(g) or len(np.unique(idx)) != len(idx):
            raise ValueError(f"{name} indices must be distinct and within 0..{len(g) - 1}")
    if np.intersect1d(fit, held).size:
        raise ValueError("fit and held-out coalitions overlap; SAR is a held-out estimate")
    n_params = 1 + masks.shape[1] + masks.shape[1] * (masks.shape[1] - 1) // 2
    if len(fit) < n_params:
        raise ValueError(f"the pairwise model has {n_params} coefficients but only {len(fit)} fitting coalitions")
    return fit, held


def sar(g, masks, fit, held, g_full=None, eps=1e-8):
    """SAR_1, SAR_2, Delta_2, rho_2 and held-out coalition metrics of the first order and pairwise fits.

    g: behavior per coalition, shape (M,); masks: (M, K) bool; fit / held: disjoint index arrays.
    g_full: behavior of the full coalition g(N), only needed for the sign error.
    Returns a dict with sar1, sar2, delta2, rho2, var (held-out variance V) and, for r = 1, 2,
    rank{r} (rank fidelity), E{r} (ordering error), regret{r} (selection regret) and sign{r} (sign error).
    """
    g, masks = np.asarray(g, dtype=np.float64), np.asarray(masks, dtype=bool)
    fit, held = _check(g, masks, fit, held, g_full)
    gh = g[held]
    V = float(np.mean((gh - gh.mean()) ** 2))
    if eps > 1e-2 * V:
        warnings.warn(f"held-out variance V = {V:.3g} is not large against eps = {eps:g}, so SAR is no longer "
                      "scale free; rescale g, for example to logit units", RuntimeWarning, stacklevel=2)
    out = {"var": V}
    for r in (1, 2):
        X = design(masks, r)
        theta, _, rank, _ = np.linalg.lstsq(X[fit], g[fit], rcond=None)
        if rank < X.shape[1]:
            warnings.warn(f"order {r} design on the fitting coalitions is rank deficient ({rank} of {X.shape[1]}); "
                          "the fit is the minimum norm solution", RuntimeWarning, stacklevel=2)
        a = X[held] @ theta
        dg, da = gh[:, None] - gh[None, :], a[:, None] - a[None, :]
        out[f"sar{r}"] = float(np.mean((gh - a) ** 2)) / (V + eps)
        out[f"rank{r}"] = _spearman(a, gh)                                          # rank fidelity
        out[f"E{r}"] = float(np.mean(dg**2 * (dg * da <= 0))) / (2 * V + 1e-12)     # ordering error
        out[f"regret{r}"] = float((gh.max() - gh[np.argmax(a)]) / (gh.max() - gh.min() + 1e-8))
        if g_full is not None:
            a_full = design(np.ones((1, masks.shape[1]), bool), r) @ theta
            out[f"sign{r}"] = float(np.mean(np.sign(gh - g_full) != np.sign(a - a_full)))
    out["delta2"] = out["sar1"] - out["sar2"]
    out["rho2"] = out["delta2"] / out["sar1"] if out["sar1"] > 1e-4 else float("nan")
    return out
