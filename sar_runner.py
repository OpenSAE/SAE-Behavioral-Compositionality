"""Run SAR on a model and SAE supplied by the researcher.

Core contract
-------------
A model adapter provides (1) an observed one-token activation and (2) scalar
behavioral readouts after *replacing that activation* with a batch of vectors.
An SAE feature source provides the K largest |z_i(h)| decoded contributions
    delta_i = z_i(h) * d_i.
For coalition S we intervene at the SAME site with
    h_tilde(S) = h - sum(delta_i for i NOT in S).
This keeps all unselected content, the SAE reconstruction residual, and the
model continuation untouched, as in the paper. The original sar.py estimator
is used without modification.

The generic interface needs NumPy only. The optional TorchSAEFeatures and
HookedTransformerAdapter need PyTorch and, for the latter, a compatible local
TransformerLens HookedTransformer (not a remote/read-only model).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from sar import coalitions, sar


@dataclass(frozen=True)
class ActivationSite:
    """Observed one-token activation, opaque replay context, optional baseline score.

    activation: one dimensional, in precisely the units expected by the SAE.
    context: anything the model adapter needs to replay from that site.
    baseline_score: optional scalar from an *unmodified* forward pass.
    """

    activation: np.ndarray
    context: Any = None
    baseline_score: float | None = None


@dataclass(frozen=True)
class SARExperiment:
    """Output metrics and the data necessary to audit/reproduce the experiment."""

    metrics: dict[str, float]
    feature_ids: np.ndarray
    contributions: np.ndarray
    masks: np.ndarray
    fit_indices: np.ndarray
    held_indices: np.ndarray
    responses: np.ndarray
    full_response: float
    activation: np.ndarray

    def as_dict(self) -> dict[str, float]:
        """Return a copy of the metric summary; raw audit arrays remain attributes."""
        return dict(self.metrics)


@runtime_checkable
class ModelAdapter(Protocol):
    def capture(self, prompt: Any) -> ActivationSite:
        """Capture the model activation at an SAE-matched site."""
        ...

    def score(self, site: ActivationSite, patched_activations: np.ndarray) -> np.ndarray:
        """Read scalar behavior for each replacement vector in shape (batch, d)."""
        ...


@runtime_checkable
class FeatureSource(Protocol):
    def top_contributions(self, activation: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (K feature indices, decoded contributions with shape (K, d))."""
        ...


class FunctionalModelAdapter:
    """Connect an arbitrary model using a capture function and a batched score function.

    Examples:
        FunctionalModelAdapter(
            capture=lambda prompt: ActivationSite(h, context=prompt),
            score=lambda site, patched: downstream(patched, site.context),
        )
    """

    def __init__(self, capture: Callable[[Any], ActivationSite],
                 score: Callable[[ActivationSite, np.ndarray], np.ndarray]):
        self._capture = capture
        self._score = score

    def capture(self, prompt: Any) -> ActivationSite:
        return self._capture(prompt)

    def score(self, site: ActivationSite, patched_activations: np.ndarray) -> np.ndarray:
        return self._score(site, patched_activations)


class FunctionalSAE:
    """Connect an SAE with a NumPy-compatible encoder and row-major decoder.

    encode_fn(h) returns one feature vector of shape (n_features,).
    decoder[i] is the direction for feature i, shape (n_features, d_model).
    The SAE bias is intentionally NOT removed: it is part of the fixed residual.
    """

    def __init__(self, encode_fn: Callable[[np.ndarray], np.ndarray], decoder: np.ndarray):
        self.encode_fn = encode_fn
        self.decoder = np.asarray(decoder)
        if self.decoder.ndim != 2 or not np.isfinite(self.decoder).all():
            raise ValueError("decoder must be a finite 2-D array, shape (features, activation_width)")

    def top_contributions(self, activation: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
        h = np.asarray(activation)
        if h.ndim != 1 or h.shape[0] != self.decoder.shape[1]:
            raise ValueError(f"activation width must match decoder columns ({self.decoder.shape[1]})")
        z = np.asarray(self.encode_fn(h))
        if z.ndim == 2 and z.shape[0] == 1:
            z = z[0]
        if z.ndim != 1 or z.size != self.decoder.shape[0]:
            raise ValueError("encoder output must match decoder rows (features)")
        if not np.isfinite(z).all():
            raise ValueError("SAE encoder returned non-finite feature activations")
        if K > z.size:
            raise ValueError(f"K={K} exceeds the number of SAE features ({z.size})")
        ids = np.argsort(-np.abs(z), kind="stable")[:K]
        delta = z[ids, None] * self.decoder[ids]
        return ids.astype(int), np.asarray(delta, dtype=np.float64)


class TorchSAEFeatures:
    """Connect an SAE Lens SAE or a custom torch SAE with .encode() and .W_dec.

    W_dec must have shape (features, d_model), exactly as in SAE Lens.
    The encoder runs at its own dtype/device; only the selected K decoder rows
    are transferred to NumPy, not the entire potentially massive dictionary.
    """

    def __init__(self, sae: Any):
        try:
            import torch
        except ImportError as e:
            raise ImportError("TorchSAEFeatures requires PyTorch") from e
        if not hasattr(sae, "encode") or not hasattr(sae, "W_dec"):
            raise TypeError("SAE must expose encode(tensor) and W_dec (features, d_model)")
        if not isinstance(sae.W_dec, torch.Tensor) or sae.W_dec.ndim != 2:
            raise TypeError("SAE.W_dec must be a rank-2 torch tensor")
        self.sae = sae

    @property
    def hook_name(self) -> str | None:
        cfg = getattr(self.sae, "cfg", None)
        meta = getattr(cfg, "metadata", None)
        return getattr(meta, "hook_name", None) or getattr(cfg, "hook_name", None)

    def top_contributions(self, activation: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
        import torch
        decoder = self.sae.W_dec
        h = np.asarray(activation)
        if h.ndim != 1 or h.shape[0] != decoder.shape[1]:
            raise ValueError(
                f"SAE decoder width {decoder.shape[1]} does not match captured activation width {h.size}. "
                "Check the base model, layer, hook and SAE checkpoint.")
        if K > decoder.shape[0]:
            raise ValueError(f"K={K} exceeds SAE dictionary width ({decoder.shape[0]})")
        x = torch.as_tensor(h, dtype=decoder.dtype, device=decoder.device).unsqueeze(0)
        with torch.inference_mode():
            z = self.sae.encode(x)
            if not isinstance(z, torch.Tensor) or z.shape != (1, decoder.shape[0]):
                raise ValueError("SAE.encode must return tensor of shape (1, n_features)")
            if not bool(torch.isfinite(z).all()):
                raise ValueError("SAE encoder returned non-finite feature activations")
            indices = torch.argsort(-z[0].abs(), stable=True)[:K]
            delta = z[0, indices, None] * decoder[indices]
            if not bool(torch.isfinite(delta).all()):
                raise ValueError("SAE decoder contributions are non-finite")
        return (indices.detach().cpu().numpy().astype(int),
                delta.detach().to(dtype=torch.float64, device="cpu").numpy())


@dataclass(frozen=True)
class LogitDifference:
    """Scalar behavior logit(target) - logit(foil) at a chosen output position.

    target and foil may be token IDs or single-token strings including their
    leading space when required (e.g. ' Emma', ' Rachel' for GPT-2).
    """

    target: int | str
    foil: int | str
    output_position: int = -1


class HookedTransformerAdapter:
    """Model adapter for local TransformerLens HookedTransformer instances.

    run_mode='full' uses temporary hooks and re-runs the full forward pass.
    run_mode='resume' uses start_at_layer to replay only the suffix, which is
    permitted ONLY for blocks.L.hook_resid_pre. It is faster for the Colab demo.
    Both modes patch one token position without discarding any other activations.
    """

    _RESID_PRE = re.compile(r"^blocks\.(\d+)\.hook_resid_pre$")

    def __init__(self, model: Any, hook_name: str, behavior: LogitDifference | Callable,
                 position: int = -1, run_mode: str = "full"):
        if run_mode not in ("full", "resume"):
            raise ValueError("run_mode must be 'full' or 'resume'")
        if not isinstance(hook_name, str) or not hook_name:
            raise ValueError("hook_name is required and must match the SAE training site")
        if run_mode == "resume" and self._RESID_PRE.fullmatch(hook_name) is None:
            raise ValueError("run_mode='resume' requires blocks.L.hook_resid_pre; use 'full' for other sites")
        self.model = model
        self.hook_name = hook_name
        self.behavior = behavior
        self.position = int(position)
        self.run_mode = run_mode

    def _score_logits(self, logits: Any) -> np.ndarray:
        import torch
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError("model must return logits of shape (batch, sequence, vocabulary)")
        if isinstance(self.behavior, LogitDifference):
            def token_id(token: int | str) -> int:
                if isinstance(token, str):
                    try:
                        return int(self.model.to_single_token(token))
                    except Exception as e:
                        raise ValueError(f"{token!r} must tokenize to exactly one token") from e
                return int(token)
            target, foil = token_id(self.behavior.target), token_id(self.behavior.foil)
            if target < 0 or foil < 0 or max(target, foil) >= logits.shape[-1]:
                raise ValueError("behavior token IDs are outside the model vocabulary")
            values = logits[:, self.behavior.output_position, target] - logits[:, self.behavior.output_position, foil]
        else:
            values = self.behavior(logits)
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        return np.asarray(values, dtype=np.float64).reshape(-1)

    def capture(self, prompt: str) -> ActivationSite:
        import torch
        if not hasattr(self.model, "run_with_cache") or not hasattr(self.model, "run_with_hooks"):
            raise TypeError("model must be a local TransformerLens model with working cache and intervention hooks")
        tokens = self.model.to_tokens(prompt)
        with torch.inference_mode():
            clean_logits, cache = self.model.run_with_cache(
                tokens, names_filter=lambda name: name == self.hook_name)
        if self.hook_name not in cache:
            raise ValueError(f"hook {self.hook_name!r} was not emitted by this model. "
                             "Check SAE and model hook names.")
        hidden = cache[self.hook_name]
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("SAR expects a residual-like hook of shape (batch=1, sequence, d_model)")
        if not -hidden.shape[1] <= self.position < hidden.shape[1]:
            raise ValueError("token position is outside the cached activation sequence")
        pos = self.position % hidden.shape[1]
        h = hidden[0, pos].detach().cpu().to(torch.float64).numpy()
        baseline = float(self._score_logits(clean_logits)[0])
        context = {"tokens": tokens.detach(), "hidden": hidden[0].detach(), "pos": pos}
        return ActivationSite(h, context=context, baseline_score=baseline)

    def score(self, site: ActivationSite, patched_activations: np.ndarray) -> np.ndarray:
        import torch
        patches = np.asarray(patched_activations)
        if patches.ndim != 2 or patches.shape[1] != site.activation.size:
            raise ValueError("patches must have shape (batch, activation_width)")
        context = site.context
        old = context["hidden"]
        pos = context["pos"]
        data = torch.as_tensor(patches, device=old.device, dtype=old.dtype)
        with torch.inference_mode():
            if self.run_mode == "resume":
                layer = int(self._RESID_PRE.fullmatch(self.hook_name).group(1))
                patched = old.unsqueeze(0).repeat(len(data), 1, 1)
                patched[:, pos, :] = data
                logits = self.model(patched, start_at_layer=layer, return_type="logits")
            else:
                tokens = context["tokens"].repeat(len(data), 1)

                def patch_fn(activations: Any, hook: Any) -> Any:
                    if activations.ndim != 3 or activations.shape[0] != len(data):
                        raise ValueError("model emitted a hook shape incompatible with SAR")
                    updated = activations.clone()
                    updated[:, pos, :] = data.to(device=updated.device, dtype=updated.dtype)
                    return updated

                logits = self.model.run_with_hooks(
                    tokens, fwd_hooks=[(self.hook_name, patch_fn)], return_type="logits")
        return self._score_logits(logits)


def _validate_coalition_data(site: ActivationSite, ids: np.ndarray,
                             contributions: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.asarray(site.activation, dtype=np.float64)
    ids = np.asarray(ids)
    delta = np.asarray(contributions, dtype=np.float64)
    if h.ndim != 1 or h.size == 0 or not np.isfinite(h).all():
        raise ValueError("captured activation must be a finite one-dimensional vector")
    if ids.shape != (K,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("feature IDs must be a 1-D integer array of length K")
    if np.any(ids < 0) or len(set(ids.tolist())) != K:
        raise ValueError("selected feature IDs must be distinct non-negative integers")
    if delta.shape != (K, h.size) or not np.isfinite(delta).all():
        raise ValueError(f"decoded contributions must be finite array of shape ({K}, {h.size})")
    return h, ids.astype(int), delta


def evaluate_sae(model: ModelAdapter, sae: FeatureSource, prompt: Any, *, K: int = 8,
                 M: int = 128, n_fit: int = 96, p: float = 0.5, seed: int = 0,
                 batch_size: int = 16, baseline_atol: float = 5e-3) -> SARExperiment:
    """Evaluate SAR on a model/SAE combination without altering the core estimator.

    model: adapter exposing capture(prompt) and score(site, patches)->(batch,).
    sae: feature source exposing top_contributions(h,K)->(ids, delta[K,d]).
    K/M/n_fit/p/seed: forwarded to the unchanged sar.coalitions() function.
    batch_size: maximum number of patched activations forwarded at a time.

    The procedure separately evaluates the original full activation and checks
    it against the unmodified model score when available; every sampled subset
    changes ONLY its selected decoded contributions.
    """
    if not isinstance(K, int) or K < 1:
        raise ValueError("K must be a positive integer")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(M, int) or not isinstance(n_fit, int):
        raise ValueError("M and n_fit must be integers")
    n_params = 1 + K + K * (K - 1) // 2
    if n_fit < n_params:
        raise ValueError(f"n_fit must be at least {n_params} for a K={K} pairwise fit")
    if not (0 < n_fit < M):
        raise ValueError("require 0 < n_fit < M")
    if not np.isfinite(baseline_atol) or baseline_atol < 0:
        raise ValueError("baseline_atol must be nonnegative and finite")
    site = model.capture(prompt)
    if not isinstance(site, ActivationSite):
        raise TypeError("model.capture(prompt) must return ActivationSite")
    if (isinstance(sae, TorchSAEFeatures) and isinstance(model, HookedTransformerAdapter)
            and sae.hook_name is not None and sae.hook_name != model.hook_name):
        raise ValueError(f"SAE was trained for hook {sae.hook_name!r}, not {model.hook_name!r}")
    h, ids, delta = _validate_coalition_data(site, *sae.top_contributions(site.activation, K), K)
    masks, fit, held = coalitions(K=K, M=M, p=p, n_fit=n_fit, seed=seed)
    patches = h[None, :] - (~masks).astype(np.float64) @ delta
    # The full coalition is mathematically identical to the original state.
    full_response = np.asarray(model.score(site, h[None, :]), dtype=np.float64)
    if full_response.shape != (1,) or not np.isfinite(full_response).all():
        raise ValueError("model.score on the unmodified activation must return one finite scalar")
    g_full = float(full_response[0])
    if site.baseline_score is not None and not np.isclose(
            g_full, site.baseline_score, rtol=5e-4, atol=baseline_atol):
        raise ValueError("replayed full coalition disagrees with the clean forward pass; "
                         "check the hook site, replay mode and model implementation")
    values = []
    for start in range(0, M, batch_size):
        batch = patches[start:start + batch_size]
        scores = np.asarray(model.score(site, batch), dtype=np.float64)
        if scores.shape != (len(batch),) or not np.isfinite(scores).all():
            raise ValueError(f"model.score must return one finite scalar per patch, got {scores.shape}")
        values.append(scores)
    responses = np.concatenate(values)
    metrics = sar(responses, masks, fit, held, g_full=g_full)
    return SARExperiment(
        metrics=metrics, feature_ids=ids.copy(), contributions=delta.copy(), masks=masks.copy(),
        fit_indices=fit.copy(), held_indices=held.copy(), responses=responses,
        full_response=g_full, activation=h.copy())
