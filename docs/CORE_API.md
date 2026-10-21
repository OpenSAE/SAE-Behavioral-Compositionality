# SAR core API

The standalone estimator is [`sar.py`](../sar.py). It accepts a scalar behavior for each binary feature coalition and does not depend on PyTorch:

```python
from sar import coalitions, sar
masks, fit, held = coalitions(K=16, M=384, n_fit=256)
# Compute g[j] by running your frozen model with exactly the features in masks[j] retained.
metrics = sar(g, masks, fit, held, g_full=full_behavior)
```

`g` has shape `(M,)`. Each Boolean row in `masks` has shape `(K,)` and indicates which selected SAE contributions are kept. The fit and held-out indices must not overlap. With an observed activation `h` and decoded contributions `delta` of shape `(K, d_model)`, the patched activations are `h - (~masks) @ delta`; unselected activation content and the SAE reconstruction residual remain fixed.

For an end-to-end integration, use [`evaluate_sae`](../sar_runner.py), as in the root [Quickstart](../README.md#-quickstart). Its public inputs are:

- `model.capture(prompt) -> ActivationSite`, containing a 1D SAE-site activation and optional replay context.
- `model.score(site, patched_activations) -> np.ndarray`, one scalar per row of patched activations.
- `sae.top_contributions(h, K) -> (feature_ids, decoded_contributions)`.

`FunctionalModelAdapter` and `FunctionalSAE` turn Python functions and NumPy arrays into these interfaces. `TorchSAEFeatures` accepts a Torch SAE with `.encode()` and `.W_dec`; `HookedTransformerAdapter` works with a matching TransformerLens model and hook. A mismatch of the SAE hook and model site is rejected.

`evaluate_sae(model, sae, prompt, K=8, M=128, n_fit=96, seed=0, batch_size=16)` returns `SARExperiment`. Read `result.metrics["sar1"]`, `["sar2"]`, `["rho2"]`, `["rank1"]`, `["rank2"]` or inspect `responses`, `masks`, `feature_ids` and the held-out indices for auditing. These default values support a small experiment; use `K=16, M=384, n_fit=256` to match the main coalition budget reported in the paper.

Details of reproducible public-model experiments and output files live in [`assets/demo/`](../assets/demo/README.md); the [Colab notebooks](../notebooks/README.md) show how to connect pretrained SAEs.
