"""Properties of the SAR estimator that the paper's conclusions rely on.

    pytest -q
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from sar import coalitions, design, rotation, sar

K = 16


def quickstart():
    masks, fit, held = coalitions(K=6)
    s = masks.astype(float)
    noise = 0.02 * np.random.default_rng(0).standard_normal(len(s))
    g = s @ [1.0, 0.5, 1.0, -0.3, 0.2, 0.1] - 1.0 * s[:, 0] * s[:, 2] + noise
    return g, masks, fit, held


def game(seed=3):
    """A K = 16 game with pairwise structure, one third order term and noise."""
    masks, fit, held = coalitions(K=K, seed=3)
    rng = np.random.default_rng(seed)
    coef = rng.normal(0, 0.3, 1 + K + K * (K - 1) // 2)
    f = lambda m: design(m, 2) @ coef + 0.4 * m[:, 0] * m[:, 1] * m[:, 2]
    g = f(masks) + 0.05 * rng.standard_normal(len(masks))
    return g, masks, fit, held, float(f(np.ones((1, K), bool))[0])


def test_quickstart_matches_readme():
    r = sar(*quickstart())
    assert (round(r["sar1"], 3), round(r["sar2"], 3), round(r["rho2"], 2)) == (0.271, 0.002, 0.99)
    assert (round(r["rank1"], 3), round(r["rank2"], 3)) == (0.850, 0.991)


@pytest.mark.parametrize("readme", ["README.md", "README_cn.md"])
def test_readme_quickstart_prints_what_it_says(readme, capsys):
    text = (Path(__file__).parents[1] / readme).read_text(encoding="utf8")
    code = re.search(r"## 🚀 [^\n]+[\s\S]*?```python\n(.*?)```", text, re.S).group(1)
    exec(compile(code, readme, "exec"), {})
    expected = [line[2:] for line in code.splitlines() if line.startswith("# ")]
    assert capsys.readouterr().out.splitlines() == expected


def test_values_are_stable():
    """Reference values from the version that matches the paper's code to 1e-13."""
    g, masks, fit, held, g_full = game()
    expected = {"var": 1.821326493568, "sar1": 0.459737238591, "rank1": 0.750211728621, "E1": 0.068251096177,
                "regret1": 0.486075766947, "sign1": 0.2109375, "sar2": 0.005123136357, "rank2": 0.997115912836,
                "E2": 6.5881307e-05, "regret2": 0.01854812132, "sign2": 0.03125, "delta2": 0.454614102233,
                "rho2": 0.988856381586}
    r = sar(g, masks, fit, held, g_full=g_full)
    for k, v in expected.items():
        assert r[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k


def test_affine_behavior_has_zero_residual():
    masks, fit, held = coalitions(K=K, seed=1)
    w = np.random.default_rng(1).normal(size=K)
    r = sar(0.7 + masks @ w, masks, fit, held)
    assert r["sar1"] < 1e-20 and r["sar2"] < 1e-20
    assert np.isnan(r["rho2"])                          # no first order residual, so no pairwise share


def test_pairwise_game_is_recovered_by_pairs():
    masks, fit, held = coalitions(K=K, seed=2)
    coef = np.random.default_rng(2).normal(size=1 + K + K * (K - 1) // 2)
    r = sar(design(masks, 2) @ coef, masks, fit, held)
    assert r["sar1"] > 0.05 and r["sar2"] < 1e-20
    assert r["rho2"] == pytest.approx(1.0) and r["rank2"] == pytest.approx(1.0)


def test_third_order_term_is_left_to_sar2():
    masks, fit, held = coalitions(K=K, seed=4)
    s = masks.astype(float)
    r = sar(s[:, 0] * s[:, 1] * s[:, 2], masks, fit, held)
    assert r["sar2"] > 0.01 and r["sar1"] > r["sar2"]


@pytest.mark.parametrize("scale", [0.1, 10.0, 1000.0])
def test_scale_free_when_variance_is_large(scale):
    g, masks, fit, held, g_full = game()                # noisy, so no exact ties for the rank metrics
    base, scaled = sar(g, masks, fit, held, g_full=g_full), sar(scale * g, masks, fit, held, g_full=scale * g_full)
    for k in ("sar1", "sar2", "rho2", "rank1", "rank2", "E1", "E2", "regret1", "regret2", "sign1", "sign2"):
        assert scaled[k] == pytest.approx(base[k], rel=1e-5), k


def test_small_variance_warns():
    g, masks, fit, held = quickstart()
    with pytest.warns(RuntimeWarning, match="scale free"):
        sar(1e-4 * g, masks, fit, held)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sar(g, masks, fit, held)                        # logit scale behaviors stay silent


def test_rotation_keeps_span_and_sum():
    for seed in range(20):
        Q = rotation(K, seed)
        assert np.allclose(Q @ Q.T, np.eye(K)) and np.allclose(Q @ np.ones(K), np.ones(K))
        delta = np.random.default_rng(seed).normal(size=(K, 8))       # K contributions in an 8-dim space
        assert np.allclose((Q.T @ delta).sum(0), delta.sum(0))         # total contribution unchanged
    assert np.array_equal(rotation(K, 5), rotation(K, 5)) and not np.allclose(rotation(K, 5), rotation(K, 6))


def test_design_columns():
    masks, _, _ = coalitions(K=5)
    assert design(masks, 1).shape == (384, 6) and design(masks, 2).shape == (384, 16)
    for order in (0, 3, "2"):
        with pytest.raises(ValueError):
            design(masks, order)


def test_sign_error_convention():
    """Three valued signs: exactly one zero side is an error, both zero is not. Same as the paper's code."""
    g, masks, fit, held, g_full = game()
    g = g.copy()
    g[held[:3]] = g_full                                # e(S) = 0 exactly for three held out coalitions
    r = sar(g, masks, fit, held, g_full=g_full)
    for order in (1, 2):
        theta = np.linalg.lstsq(design(masks, order)[fit], g[fit], rcond=None)[0]
        e = g[held] - g_full
        e_hat = design(masks, order)[held] @ theta - design(np.ones((1, K), bool), order)[0] @ theta
        assert r[f"sign{order}"] == np.mean(np.sign(e) != np.sign(e_hat))
        one_zero = np.sum((e == 0) != (e_hat == 0))
        assert r[f"sign{order}"] * len(held) == np.sum(e * e_hat < 0) + one_zero


@pytest.mark.parametrize("change", ["overlap", "nan", "inf", "shape", "duplicate", "range", "few_fit"])
def test_bad_input_is_rejected(change):
    g, masks, fit, held = quickstart()
    if change == "overlap":
        held = np.concatenate([held, fit[:1]])
    elif change == "nan":
        g = g.copy(); g[0] = np.nan
    elif change == "inf":
        g = g.copy(); g[0] = np.inf
    elif change == "shape":
        g = g[:-1]
    elif change == "duplicate":
        held = np.concatenate([held, held[:1]])
    elif change == "range":
        held = np.concatenate([held, [len(g)]])
    elif change == "few_fit":
        fit = fit[:10]                                  # the pairwise model for K = 6 has 22 coefficients
    with pytest.raises(ValueError):
        sar(g, masks, fit, held)
