# 🧩 Behavioral Compositionality of Sparse Autoencoder Representations

🌍 Languages: [English](README.md) | [中文](README_cn.md)

👉 TL;DR. SAE features are decoded linearly, but their effects on model behavior may depend on mutual interactions. We introduce **SAE Additivity Residuals (SAR)** to measure the behavioral variation left unexplained by individual feature effects.

<p align="center">
  <a href="https://arxiv.org/abs/2610.21261"><b>📄 Paper (arXiv)</b></a> &nbsp; | &nbsp;
  <a href="https://opensae.github.io/SAE-Behavioral-Compositionality/"><b>🎞️ 5-minute tutorial</b></a> &nbsp; | &nbsp;
  <a href="assets/demo/README.md"><b>🧪 Examples</b></a>
</p>

## 🎥 Overview

<p align="center">
  <img src="assets/figures/fig1.png" alt="Figure 1" width="900"/>
</p>

**Figure 1.** Behavioral compositionality in SAE coordinates. (a) Selected decoded contributions are retained or
removed while all other activation content is fixed, and the patched activation is propagated through the frozen upper
layers to obtain $`g_x(S)`$. (b, c) Nearly additive and interacting responses, showing the first order fit, pairwise
interactions, and variance decomposition. $`\mathrm{SAR}_1`$ is the residual beyond individual features and
$`\rho_2=\Delta_2/\mathrm{SAR}_1`$ its pairwise share.

---

## 🎞️ 5-Minute Tutorial

<p align="center">
  <a href="https://opensae.github.io/SAE-Behavioral-Compositionality/"><img src="assets/tour_preview.png" alt="5-minute tutorial" width="900"/></a>
</p>

💻 Click to open the [tutorial](https://opensae.github.io/SAE-Behavioral-Compositionality/) in your browser

☁️ **Try SAR in Colab:** [Interactive quickstart](https://colab.research.google.com/github/OpenSAE/SAE-Behavioral-Compositionality/blob/main/notebooks/SAR_Quickstart.ipynb) · [Use your own SAE](https://colab.research.google.com/github/OpenSAE/SAE-Behavioral-Compositionality/blob/main/notebooks/Use_Your_Own_SAE.ipynb)

---

## ✨ Highlights

- 🧮 **An exact interaction decomposition.** Under independent feature subset sampling, $`\mathrm{SAR}_r`$ equals the
  normalized energy of interactions above order $`r`$. It also bounds coalition ordering error for the best order $`r`$
  approximation.
- 📏 **Affine controls.** Affine readouts give $`\widehat{\mathrm{SAR}}_1`$ at numerical zero, at most $`1.55\times10^{-13}`$.
  Real behaviors give a positive $`\widehat{\mathrm{SAR}}_1`$ in all 57 model, layer and behavior combinations.
- 🧭 **SAE coordinates matter.** Learned SAE coordinates have lower first order residuals than sum preserving random
  rotations of the same subspace in all 18 model and behavior settings.
- 🤝 **Pairs recover missing structure.** Pairwise terms raise the average held out rank fidelity from 0.9782 to 0.9959.
  They also outperform parameter matched models with random third order terms in all 18 settings.
- 🎛️ **Steering.** For 16 published SAE steering features in Llama-3.1 8B, pairwise modeling raises rank fidelity on
  MMLU-Pro from 0.5874 to 0.7790.
- 🔬 **Beyond reconstruction.** Across six matched SAEs, # 🧩 Behavioral Compositionality of Sparse Autoencoder Representations

🌍 Languages: [English](README.md) | [中文](README_cn.md)

👉 TL;DR. SAE features are decoded linearly, but their effects on model behavior may depend on mutual interactions. We introduce **SAE Additivity Residuals (SAR)** to measure the behavioral variation left unexplained by individual feature effects.

<p align="center">
  <a href="https://arxiv.org/abs/2610.21261"><b>📄 Paper (arXiv)</b></a> &nbsp; | &nbsp;
  <a href="https://opensae.github.io/SAE-Behavioral-Compositionality/"><b>🎞️ 5-minute tutorial</b></a> &nbsp; | &nbsp;
  <a href="assets/demo/README.md"><b>🧪 Examples</b></a>
</p>

## 🎥 Overview

<p align="center">
  <img src="assets/figures/fig1.png" alt="Figure 1" width="900"/>
</p>

**Figure 1.** Behavioral compositionality in SAE coordinates. (a) Selected decoded contributions are retained or
removed while all other activation content is fixed, and the patched activation is propagated through the frozen upper
layers to obtain $`g_x(S)`$. (b, c) Nearly additive and interacting responses, showing the first order fit, pairwise
interactions, and variance decomposition. $`\mathrm{SAR}_1`$ is the residual beyond individual features and
$`\rho_2=\Delta_2/\mathrm{SAR}_1`$ its pairwise share.

---

## 🎞️ 5-Minute Tutorial

<p align="center">
  <a href="https://opensae.github.io/SAE-Behavioral-Compositionality/"><img src="assets/tour_preview.png" alt="5-minute tutorial" width="900"/></a>
</p>

💻 Click to open the [tutorial](https://opensae.github.io/SAE-Behavioral-Compositionality/) in your browser

☁️ **Try SAR in Colab:** [Interactive quickstart](https://colab.research.google.com/github/OpenSAE/SAE-Behavioral-Compositionality/blob/main/notebooks/SAR_Quickstart.ipynb) · [Use your own SAE](https://colab.research.google.com/github/OpenSAE/SAE-Behavioral-Compositionality/blob/main/notebooks/Use_Your_Own_SAE.ipynb)

---

## ✨ Highlights

- 🧮 **An exact interaction decomposition.** Under independent feature subset sampling, $`\mathrm{SAR}_r`$ equals the
  normalized energy of interactions above order $`r`$. It also bounds coalition ordering error for the best order $`r`$
  approximation.
- 📏 **Affine controls.** Affine readouts give $`\widehat{\mathrm{SAR}}_1`$ at numerical zero, at most $`1.55\times10^{-13}`$.
  Real behaviors give a positive $`\widehat{\mathrm{SAR}}_1`$ in all 57 model, layer and behavior combinations.
- 🧭 **SAE coordinates matter.** Learned SAE coordinates have lower first order residuals than sum preserving random
  rotations of the same subspace in all 18 model and behavior settings.
- 🤝 **Pairs recover missing structure.** Pairwise terms raise the average held out rank fidelity from 0.9782 to 0.9959.
  They also outperform parameter matched models with random third order terms in all 18 settings.
- 🎛️ **Steering.** For 16 published SAE steering features in Llama-3.1 8B, pairwise modeling raises rank fidelity on
  MMLU-Pro from 0.5874 to 0.7790.
- 🔬 **Beyond reconstruction.** \widehat{\mathrm{SAR}}_1`$ is more strongly associated with
  sign error and selection regret than reconstruction error.
- ⚡ **Small code.** The estimator is one NumPy file, [`sar.py`](sar.py). The GPT-2 demo runs in about a minute on a GPU.

---

## 📐 What SAR Measures

<details>
<summary>🎯 The intervention.</summary>

<br>

For a prompt $`x`$, let $`h_x`$ be the activation at a chosen layer and token. An SAE writes it with
feature activations $`z_i(x)`$ and decoder directions $`d_i`$. Let $`N_x`$ hold the $`K`$ most active features, with decoded
contributions $`\delta_i(x)=z_i(x)d_i`$. Keeping a subset $`S\subseteq N_x`$ gives

$$\tilde h_x(S)=h_x^0+\sum_{i\in S}\delta_i(x),$$

where $`h_x^0`$ contains all other activation content, including the reconstruction residual. The patched activation runs
through the frozen upper layers, and a scalar behavior $`\psi`$ such as a logit difference gives $`g_x(S)`$.

</details>

<details>
<summary>🧩 SAE Additivity Residuals.</summary>

<br>

Subsets are drawn by keeping each feature independently with probability $`1/2`$. Let
$`\Pi_r g_x`$ be the best approximation of $`g_x`$ that uses terms of at most $`r`$ features, and let $`V`$ be the variance of
$`g_x`$. Then

$$\mathrm{SAR}_r=\frac{\lVert g_x-\Pi_r g_x\rVert_\mu^2}{V},\qquad \Delta_2=\mathrm{SAR}_1-\mathrm{SAR}_2,\qquad \rho_2=\frac{\Delta_2}{\mathrm{SAR}_1}.$$

$`\mathrm{SAR}_1`$ is the share of behavioral variance not explained by individual feature effects. $`\rho_2`$ is the part
of that share recovered by feature pairs.

</details>

<details>
<summary>🎲 Estimation.</summary>

<br>

With $`K=16`$ we sample 384 subsets per prompt. We fit the order $`r`$ model by least squares on 256 of
them and evaluate it on the other 128. With $`\widehat L_r`$ the held out mean squared error and $`\widehat V`$ the held out
variance of $`g_x`$, the reported estimate is $`\widehat{\mathrm{SAR}}_r=\widehat L_r/(\widehat V+\epsilon)`$.

</details>

<details>
<summary>✅ Explanation fidelity.</summary>

<br>

On the held out subsets $`\mathcal U`$ we also measure how well a fitted model $`\widehat a_r`$
orders and selects subsets.

| Metric | Definition | Meaning |
|---|---|---|
| Rank fidelity | $`F^{(r)}_{\mathrm{rank}}=\mathrm{Spearman}_{S\in\mathcal U}\big(\widehat a_r(S),g_x(S)\big)`$ | 1 means the model orders all held out subsets correctly |
| Ordering error | $`\mathcal E(a)=\mathbb E\big[(\Delta g)^2\,\mathbf 1\{\Delta g\,\Delta a\le 0\}\big]/(2V)`$ | Wrongly ordered pairs of subsets, weighted by their behavior difference |
| Sign error | Share of $`S\in\mathcal U`$ with $`\mathrm{sign}\,e(S)\ne\mathrm{sign}\,\widehat e_r(S)`$, where $`e(S)=g_x(S)-g_x(N_x)`$ and $`\widehat e_r(S)=\widehat a_r(S)-\widehat a_r(N_x)`$ | The model gets the direction of the change wrong |
| Selection regret | $`\big(g_x(S^\star)-g_x(\widehat S_r)\big)/\big(\max_{\mathcal U}g_x-\min_{\mathcal U}g_x\big)`$ | Loss from picking the subset the model ranks first |

Here $`\Delta g=g_x(S)-g_x(T)`$ and $`\Delta a=a(S)-a(T)`$ for two independent subsets $`S`$ and $`T`$, $`S^\star`$ is the best
held out subset, and $`\widehat S_r`$ is the subset the model ranks first. The paper shows
$`\mathcal E(\Pi_r g_x)\le\mathrm{SAR}_r`$.

</details>

<details>
<summary>🔢 Numerical conventions.</summary>

<br>

`sar.py` keeps the conventions of the code behind the paper, so its numbers match the
paper. The estimate uses $`\epsilon=10^{-8}`$. This is negligible for behaviors on the logit scale, but it makes
$`\widehat{\mathrm{SAR}}_r`$ depend on scale when $`\widehat V`$ is tiny, so `sar` warns when $`\epsilon`$ exceeds 1% of
$`\widehat V`$. In 4,900 prompts from the paper's experiments, $`\epsilon`$ changes the estimate by at most 1.4%.
$`\widehat\rho_2`$ is left undefined when $`\widehat{\mathrm{SAR}}_1\le10^{-4}`$. Signs are compared exactly, so a zero
change on only one side counts as a sign error.

</details>

<details>
<summary>🔄 Rotated coordinates.</summary>

<br>

To test whether compositionality depends on the SAE basis, we replace the contributions
$`D_x=[\delta_1,\dots,\delta_K]`$ by $`D_xQ`$, where $`Q`$ is a random orthogonal matrix with $`Q\mathbf 1=\mathbf 1`$. This
keeps the selected subspace, the total contribution, and the upper layers unchanged.

</details>

---

## 🔎 Comparison of Methods

<details>
<summary>🔗 Explaining a combination of features.</summary>

<br>

A common way to explain what several features do together is to add up what
each one does alone. This is the first order model. We compare it with the pairwise model and with a control that adds
the same number of random terms of three features. All models are fit on the same 256 subsets and judged on the same
128 held out subsets. Numbers are averages over 18 model and behavior settings.

| Model | Terms for $`K=16`$ | Held out rank fidelity | Ordering error |
|---|:---:|:---:|:---:|
| First order, one effect per feature | 17 | 0.9782 | baseline |
| 🏆 **Pairwise, one effect per feature and per pair** | 137 | ✅ **0.9959** | ✅ lower in 18 of 18 settings |
| Random third order control | 137 | ❌ below pairwise in 18 of 18 settings | ❌ above pairwise in 18 of 18 settings |

</details>

<details>
<summary>⚖️ Evaluating an SAE.</summary>

<br>

Reconstruction error measures how accurately an SAE represents model activations, but it does not directly
measure whether decoded features have additive downstream effects. For six SAEs at the same Gemma-3 1B
site, we compare how reconstruction error and $`\widehat{\mathrm{SAR}}_1`$ relate to two failures of first order
explanations. The numbers are prompt level Spearman correlations over three behaviors.

| Score | What it measures | With sign error | With selection regret |
|---|---|:---:|:---:|
| Reconstruction error $`\mathrm{FVU}_x`$ | How well the SAE rebuilds the activation | ❌ −0.114 to −0.013 | ❌ 0.005 to 0.028 |
| 🏆 $`\widehat{\mathrm{SAR}}_1`$ | Behavior left unexplained by individual feature effects | ✅ **0.269 to 0.430** | ✅ **0.199 to 0.235** |

</details>

---

## 🚀 Quickstart

Clone the repository and install the NumPy core:

```bash
git clone https://github.com/OpenSAE/SAE-Behavioral-Compositionality.git
cd SAE-Behavioral-Compositionality
python -m pip install numpy
```

Provide the activation at an SAE site, an SAE encoder and decoder, and a function that scores patched activations:

```python
import numpy as np
from sar_runner import ActivationSite, FunctionalModelAdapter, FunctionalSAE, evaluate_sae

h = np.array([1.5, 1.0, 0.8, 0.6, 0.4, 0.3])
model = FunctionalModelAdapter(
    capture=lambda prompt: ActivationSite(activation=h),
    score=lambda site, patched: patched.sum(axis=1) + 1.1 * patched[:, 0] * patched[:, 2],
)
sae = FunctionalSAE(encode_fn=lambda x: x, decoder=np.eye(6))
result = evaluate_sae(model, sae, prompt="example", K=6, M=384, n_fit=256)
print("Nonadditivity detected:", result.metrics["sar1"] > 0.01)
print("Pairs explain interactions:", result.metrics["sar2"] < 1e-16)
# Nonadditivity detected: True
# Pairs explain interactions: True
```

Replace the example activation, encoder and decoder with your model's SAE site. The scorer must return one scalar per patched activation. `evaluate_sae` selects the top SAE contributions, evaluates held-out coalitions and returns first-order and pairwise residuals. The estimator alone is `sar(g, masks, fit, held)` in [`sar.py`](sar.py).

[Core API and real-model adapters](docs/CORE_API.md) · [Runnable examples](assets/demo/README.md) · [Colab notebooks](notebooks/README.md)

---

## 📊 Main Results

<details>
<summary>🧠 Figure 2. Behavioral compositionality varies across behaviors and representations</summary>

<br>

<p align="center">
  <img src="assets/figures/fig2.png" alt="Figure 2" width="900"/>
</p>

Prompt median $`\widehat{\mathrm{SAR}}_1`$ with 1,000 prompts for each combination.

| | Simple copying | Subject verb agreement | Gendered pronoun | CounterFact | IOI | Greater than |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Gemma-3 4B, layer 22 | 0.03668 | 0.03732 | 0.03532 | 0.05275 | 0.04174 | 0.02121 |
| Mistral-7B, layer 24 | 0.02029 | 0.02435 | 0.04939 | 0.05535 | 0.01510 | 0.04159 |
| Llama-3.1 8B, layer 21 | 0.04486 | 0.03070 | 0.08244 | 0.07707 | 0.04245 | 0.05625 |

$`\widehat{\mathrm{SAR}}_1`$ is positive in all 57 model, layer and behavior combinations, while affine controls stay at
numerical zero. Its size depends on both the behavior and the model.

</details>

<details>
<summary>🧭 Figure 3. Compositionality depends on coordinate choice</summary>

<br>

<p align="center">
  <img src="assets/figures/fig3.png" alt="Figure 3" width="900"/>
</p>

Median $`\widehat{\mathrm{SAR}}_1`$ for SAE coordinates and for random rotations, with 100 prompts and eight rotations
per prompt.

| SAE / rotation | Simple copying | Subject verb agreement | Gendered pronoun | CounterFact | IOI | Greater than |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Gemma-3 4B | **0.041** / 0.083 | **0.033** / 0.076 | **0.039** / 0.081 | **0.056** / 0.094 | **0.021** / 0.024 | **0.022** / 0.032 |
| Mistral-7B | **0.023** / 0.058 | **0.022** / 0.120 | **0.024** / 0.141 | **0.055** / 0.133 | **0.016** / 0.038 | **0.065** / 0.066 |
| Llama-3.1 8B | **0.059** / 0.166 | **0.019** / 0.070 | **0.109** / 0.149 | **0.080** / 0.138 | **0.049** / 0.079 | **0.063** / 0.178 |

The selected subspace, the total contribution and the upper layers are the same in each pair. Only the coordinates
change, and the SAE coordinates have the lower residual in all 18 settings.

</details>

<details>
<summary>📉 Figure 4. Estimated SAR is associated with first order coalition fidelity</summary>

<br>

<p align="center">
  <img src="assets/figures/fig4.png" alt="Figure 4" width="900"/>
</p>

From the lowest to the highest $`\widehat{\mathrm{SAR}}_1`$ decile, first order rank fidelity falls from 0.9857 to 0.9539
in Gemma-3 4B, from 0.9939 to 0.9413 in Mistral-7B, and from 0.9790 to 0.9289 in Llama-3.1 8B. Selection regret
roughly doubles in Gemma and Llama and grows about fivefold in Mistral.

</details>

<details>
<summary>🤝 Figure 5. Feature pairs recover missing coalition structure</summary>

<br>

<p align="center">
  <img src="assets/figures/fig5.png" alt="Figure 5" width="900"/>
</p>

| Average over 18 settings | First order | Pairwise |
|---|:---:|:---:|
| Rank fidelity | 0.9782 | **0.9959** |
| Selection regret | 0.0272 | **0.0051** |

Pairwise models also lower ordering error and sign error in all 18 settings, and the gains hold in all 20 repeated
splits of the subsets.

</details>

<details>
<summary>🪜 Figures 6 and 7. Estimated compositionality varies across depth</summary>

<br>

<p align="center">
  <img src="assets/figures/fig6.png" alt="Figure 6" width="900"/>
</p>

<p align="center">
  <img src="assets/figures/fig7.png" alt="Figure 7" width="900"/>
</p>

Simple copying at the earliest and latest sampled layers.

| | GPT-2 Small | Pythia-70M | Gemma-3 270M | Gemma-3 1B | Gemma-3 4B | Mistral-7B | Llama-3.1 8B |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $`\widehat{\mathrm{SAR}}_1`$ | 0.298 → 0.013 | 0.137 → 0.0007 | 0.174 → 0.007 | 0.154 → 0.015 | 0.112 → 0.012 | 0.177 → 0.020 | 0.200 → 0.256 |
| $`\widehat\rho_2`$ | 0.34 → 0.95 | 0.59 → 1.00 | 0.70 → 0.99 | 0.70 → 0.98 | 0.58 → 0.71 | 0.53 → 0.79 | 0.54 → 0.78 |

Across the nine early to late comparisons of Figure 6, $`\widehat{\mathrm{SAR}}_1`$ decreases in eight and $`\widehat\rho_2`$
increases in all nine. SAE coordinates have lower $`\widehat{\mathrm{SAR}}_1`$ than rotations in all 33 model, behavior
and layer comparisons.

</details>

<details>
<summary>🎛️ Table 1. Pairwise modeling better predicts the aggregate steering response</summary>

<br>

Sixteen published CorrSteer features in Llama-3.1 8B, 1,024 steering combinations, and 1,400 MMLU-Pro questions.

| Approximation | Residual estimate | Rank fidelity | Ordering error |
|---|:---:|:---:|:---:|
| First order | 0.6424 | 0.5874 | 0.1444 |
| Pairwise | **0.3891** | **0.7790** | **0.0606** |

</details>

<details>
<summary>🔬 Table 2 and Figure 8. Reconstruction quality and SAR differ in a matched SAE comparison</summary>

<br>

<p align="center">
  <img src="assets/figures/fig8.png" alt="Figure 8" width="900"/>
</p>

Prompt level Spearman correlations pooled over six SAEs at Gemma-3 1B layer 17, with 1,800 prompts per behavior.

| Behavior | $`\mathrm{FVU}_x`$ and sign error | $`\widehat{\mathrm{SAR}}_1`$ and sign error | $`\mathrm{FVU}_x`$ and selection regret | $`\widehat{\mathrm{SAR}}_1`$ and selection regret |
|---|:---:|:---:|:---:|:---:|
| Simple copying | −0.080 | **0.269** | 0.011 | **0.220** |
| Subject verb agreement | −0.114 | **0.430** | 0.005 | **0.235** |
| IOI | −0.013 | **0.396** | 0.028 | **0.199** |

</details>

---

## 🗺️ Figure Map

<details>
<summary>📋 Figure index and captions</summary>

<br>

Captions are taken from the paper.

| Figure | Caption | In the demo |
|---|---|:---:|
| [Figure&nbsp;1](assets/figures/fig1.png) | Behavioral compositionality in SAE coordinates. | ✅ |
| [Figure&nbsp;2](assets/figures/fig2.png) | Prompt median $`\widehat{\mathrm{SAR}}_1`$ for six behaviors in Gemma-3 4B layer 22, Mistral-7B layer 24, and Llama-3.1 8B layer 21. Error bars show 95% prompt bootstrap intervals. | |
| [Figure&nbsp;3](assets/figures/fig3.png) | Paired first order SAR for learned SAE coordinates and sum preserving random rotations of the same selected activation subspace. | ✅ panel A |
| [Figure&nbsp;4](assets/figures/fig4.png) | First order explanation fidelity across $`\widehat{\mathrm{SAR}}_1`$ deciles within each model. | |
| [Figure&nbsp;5](assets/figures/fig5.png) | Held out coalition ordering for first order and pairwise approximations. | ✅ panel D |
| [Figure&nbsp;6](assets/figures/fig6.png) | Depth profiles for simple copying, CounterFact, and IOI. | ✅ panels B and C |
| [Figure&nbsp;7](assets/figures/fig7.png) | Simple copying at the earliest and latest sampled layers across seven model and SAE pipelines. | ✅ GPT-2 row |
| [Figure&nbsp;8](assets/figures/fig8.png) | Median prompt level $`\mathrm{FVU}_x`$ and median $`\widehat{\mathrm{SAR}}_1`$ for six SAEs at Gemma-3 1B layer 17. | |
| [Figure&nbsp;9](assets/figures/fig9.png) | Behavior scores for four examples under ablations of individual features and feature pairs, together with first order and pairwise predictions when available. | |

</details>

<details>
<summary>Show all figures</summary>

<br>

| | |
|---|---|
| <img src="assets/figures/fig2.png" width="420"/><br>Figure 2 | <img src="assets/figures/fig3.png" width="420"/><br>Figure 3 |
| <img src="assets/figures/fig4.png" width="420"/><br>Figure 4 | <img src="assets/figures/fig5.png" width="420"/><br>Figure 5 |
| <img src="assets/figures/fig6.png" width="420"/><br>Figure 6 | <img src="assets/figures/fig7.png" width="420"/><br>Figure 7 |
| <img src="assets/figures/fig8.png" width="420"/><br>Figure 8 | <img src="assets/figures/fig9.png" width="420"/><br>Figure 9 |

</details>

---

## 🙏 Appreciation

This project builds on [SAE Lens](https://github.com/decoderesearch/SAELens),
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [Neuronpedia](https://www.neuronpedia.org).
It uses the open SAEs for [GPT-2 Small](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted),
[Pythia-70M](https://huggingface.co/ctigges/pythia-70m-deduped__res-sm_processed),
[Gemma Scope 2](https://huggingface.co/google/gemma-scope-2-4b-pt),
[Mistral-7B](https://huggingface.co/JoshEngels/Mistral-7B-Residual-Stream-SAEs) and
[Llama Scope](https://huggingface.co/OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x), the
[CounterFact](https://rome.baulab.info), [IOI](https://arxiv.org/abs/2211.00593) and
[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) data, and the
[CorrSteer](https://arxiv.org/abs/2508.12535) steering features.

---

## 📚 Citation

If you use this work, please cite the paper:

```bibtex
@misc{ma2026behavioral,
  title = {Behavioral Compositionality of Sparse Autoencoder Representations},
  author = {Haotian Ma and Ziye Mei and Frederic Sala},
  year = {2026},
  eprint = {2610.21261},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2610.21261}
}
```
