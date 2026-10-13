"""New adapter tests; original tests/test_sar.py and sar.py stay unchanged."""
import numpy as np
import pytest

from sar import coalitions, sar
from sar_runner import (
    ActivationSite, FunctionalModelAdapter, FunctionalSAE,
    HookedTransformerAdapter, LogitDifference, TorchSAEFeatures, evaluate_sae,
)


def numpy_problem():
    h = np.array([1.5, 1.0, 0.8, 0.6, 0.4, 0.3], dtype=float)
    # Two features interact; the remainder is kept by constructing from h.
    model = FunctionalModelAdapter(
        capture=lambda p: ActivationSite(h, context=p, baseline_score=float(h.sum() + 1.1*h[0]*h[2])),
        score=lambda site, a: a.sum(1) + 1.1*a[:, 0]*a[:, 2],
    )
    sae = FunctionalSAE(lambda a: a, np.eye(len(h)))
    return model, sae, h


def test_generic_sae_matches_manual_interventions_and_original_estimator():
    model, sae, h = numpy_problem()
    result = evaluate_sae(model, sae, prompt="any", K=6, M=384, n_fit=256, seed=13, batch_size=17)
    masks, fit, held = coalitions(K=6, M=384, n_fit=256, seed=13)
    delta = h[:, None] * np.eye(len(h))
    patched = h - (~masks) @ delta
    g = patched.sum(1) + 1.1 * patched[:, 0] * patched[:, 2]
    direct = sar(g, masks, fit, held, g_full=h.sum() + 1.1*h[0]*h[2])
    assert np.array_equal(result.feature_ids, np.arange(6))
    assert np.array_equal(result.masks, masks)
    assert np.array_equal(result.fit_indices, fit)
    assert np.array_equal(result.held_indices, held)
    assert np.allclose(result.responses, g)
    for key in ("sar1", "sar2", "rho2", "rank1", "rank2", "E1", "E2", "sign1", "sign2"):
        assert result.metrics[key] == pytest.approx(direct[key], abs=1e-10), key
    assert result.metrics["sar1"] > 0.01
    assert result.metrics["sar2"] < 1e-18
    assert result.metrics["rho2"] == pytest.approx(1)
    assert result.metrics["rank2"] > 0.999


def test_nonzero_unselected_residual_kept_exactly():
    h = np.array([1.0, 2.0, 4.0])
    # The SAE explains only coordinate 0; the residual [0,2,4] is fixed.
    sae = FunctionalSAE(lambda a: np.array([a[0]]), np.array([[1., 0., 0.]]))
    model = FunctionalModelAdapter(lambda _: ActivationSite(h), lambda _, a: a[:, 0] + 2*a[:, 1] + 4*a[:, 2])
    result = evaluate_sae(model, sae, "p", K=1, M=30, n_fit=20, batch_size=5)
    assert result.full_response == 21
    assert set(result.responses) == {20., 21.}
    assert np.all(result.contributions == [[1., 0., 0.]])
    assert result.metrics["sar1"] < 1e-20


def test_custom_sae_uses_top_k_and_stable_tie_break():
    x = np.array([0.1, -2., 2.])
    sae = FunctionalSAE(lambda a: a, np.eye(3))
    idx, delta = sae.top_contributions(x, K=2)
    assert idx.tolist() == [1, 2]
    assert np.array_equal(delta, np.array([[0., -2., 0.], [0., 0., 2.]]))


@pytest.mark.parametrize("bad", ["bad_decoder", "width", "features", "nan", "large_k"])
def test_misconfigured_sae_errors(bad):
    if bad == "bad_decoder":
        with pytest.raises(ValueError, match="2-D"):
            FunctionalSAE(lambda x: x, np.ones(3))
        return
    enc = lambda x: x
    decoder = np.eye(3)
    if bad == "width":
        decoder = np.ones((3, 4))
    elif bad == "features":
        enc = lambda x: np.ones(2)
    elif bad == "nan":
        enc = lambda x: np.array([1, np.nan, 2])
    sae = FunctionalSAE(enc, decoder)
    with pytest.raises(ValueError):
        sae.top_contributions(np.array([0.2, 0.3, 0.4]), 4 if bad == "large_k" else 2)


def test_bad_replay_is_caught():
    m, s, h = numpy_problem()
    m = FunctionalModelAdapter(
        lambda _: ActivationSite(h, baseline_score=-999),
        lambda _, a: a.sum(1) + 1.1*a[:, 0]*a[:, 2])
    with pytest.raises(ValueError, match="disagrees"):
        evaluate_sae(m, s, "p", K=6, M=64, n_fit=46)


def test_invalid_config_and_model_return_shape_are_rejected():
    m, s, h = numpy_problem()
    for kwargs in ({"K": 0}, {"K": 6, "n_fit": 10}, {"K": 6, "batch_size": 0}, {"K": 6, "M": 12}):
        with pytest.raises(ValueError):
            evaluate_sae(m, s, "p", **kwargs)
    bad = FunctionalModelAdapter(lambda _: ActivationSite(h), lambda _, a: np.zeros((len(a), 1)))
    with pytest.raises(ValueError, match="must return one"):
        evaluate_sae(bad, s, "p", K=6, M=64, n_fit=46)


def test_torch_sae_features_only_moves_selected_rows():
    torch = pytest.importorskip("torch")

    class MySAE:
        W_dec = torch.eye(6, dtype=torch.float32)
        def encode(self, x):
            return torch.relu(x)

    x = np.array([1., 2., -3., 4., 5., 6.])
    f = TorchSAEFeatures(MySAE())
    ids, delta = f.top_contributions(x, 3)
    assert ids.tolist() == [5, 4, 3]
    assert np.allclose(delta, np.array([[0., 0., 0., 0., 0., 6.],
                                        [0., 0., 0., 0., 5., 0.],
                                        [0., 0., 0., 4., 0., 0.]]))
    with pytest.raises(ValueError, match="does not match"):
        f.top_contributions(np.ones(7), 2)


def test_hooked_transformer_both_intervention_modes_and_clean_baseline():
    torch = pytest.importorskip("torch")

    class ToyHookedTransformer:
        def to_tokens(self, prompt):
            return torch.tensor([[1, 2, 3]])
        def to_single_token(self, t):
            return {" target": 0, " foil": 1}[t]
        def _hidden(self, tokens):
            h = torch.tensor([1., 2., 3., 4., 5., 6.], dtype=torch.float32)
            return h.view(1, 1, 6).expand(len(tokens), tokens.shape[1], -1).clone()
        def _logits(self, states):
            z = states[..., 0] + 0.3*states[..., 1]*states[..., 3] + states[..., 2]
            return torch.stack([z, torch.zeros_like(z)], dim=-1)
        def run_with_cache(self, tokens, names_filter):
            h = self._hidden(tokens)
            return self._logits(h), {"blocks.2.hook_resid_pre": h}
        def run_with_hooks(self, tokens, fwd_hooks, return_type):
            h = self._hidden(tokens)
            for name, fn in fwd_hooks:
                assert name == "blocks.2.hook_resid_pre"
                h = fn(h, None)
            return self._logits(h)
        def __call__(self, patch, *, start_at_layer, return_type):
            assert start_at_layer == 2
            return self._logits(patch)

    class ToySAE:
        W_dec = torch.eye(6)
        class cfg:
            class metadata:
                hook_name = "blocks.2.hook_resid_pre"
        def encode(self, x):
            return x

    m = ToyHookedTransformer()
    sae = TorchSAEFeatures(ToySAE())
    results = []
    for mode in ("full", "resume"):
        runner = HookedTransformerAdapter(m, "blocks.2.hook_resid_pre",
                                           LogitDifference(" target", " foil"), run_mode=mode)
        site = runner.capture("some tokens")
        assert site.activation.shape == (6,)
        scores = runner.score(site, site.activation.reshape(1, -1))
        assert scores[0] == pytest.approx(site.baseline_score)
        r = evaluate_sae(runner, sae, "test", K=6, M=100, n_fit=65, batch_size=13)
        assert np.isfinite(r.metrics["sar1"])
        results.append(r)
    assert np.allclose(results[0].responses, results[1].responses)
    assert results[0].metrics["sar1"] == pytest.approx(results[1].metrics["sar1"])
    with pytest.raises(ValueError, match="requires"):
        HookedTransformerAdapter(m, "blocks.2.hook_resid_post",
                                 LogitDifference(0, 1), run_mode="resume")
    sae.sae.cfg.metadata.hook_name = "blocks.4.hook_resid_pre"
    with pytest.raises(ValueError, match="trained for hook"):
        evaluate_sae(runner, sae, "p", K=6, M=100, n_fit=65)
