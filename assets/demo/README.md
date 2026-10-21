# SAR examples

Runnable demonstrations are kept here, separate from the public estimator [`sar.py`](../../sar.py) and model adapter [`sar_runner.py`](../../sar_runner.py).

Install the optional model dependencies from the repository root:

```bash
python -m pip install -r requirements_demo.txt
```

Run the public GPT-2 Small SAE experiment from the repository root:

```bash
python assets/demo/demo.py
python assets/demo/demo.py --layers 4 6 8 10
```

The script evaluates 24 copying prompts, 16 SAE contributions and 384 coalitions per prompt, comparing SAE and sum-preserving rotated coordinates. It prints first-order and pairwise held-out fidelity, then saves a demonstration image in the current working directory as `demo_layers4_6_8_10.png` when the four layers are selected.

The bundled [reference image](demo_layers4_6_8_10.png) is a recorded demonstration result, not the paper's complete evaluation. The current model or library versions may produce slightly different numbers. Notebook examples for custom SAE checkpoints are in [`notebooks/`](../../notebooks/README.md).

For embedding SAR in another project, use the single [Quickstart](../../README.md#-quickstart) or the [core API contract](../../docs/CORE_API.md).
