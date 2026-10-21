# SAR Colab notebooks

These notebooks extend SAR without changing the original `sar.py`, `assets/demo/demo.py`, paper results, or tutorial website.

| Notebook | Best for | Open |
|---|---|---|
| [SAR Quickstart](SAR_Quickstart.ipynb) | First-time users; transparent game then real GPT-2 + public SAE | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenSAE/SAE-Behavioral-Compositionality/blob/main/notebooks/SAR_Quickstart.ipynb) |
| [Use Your Own SAE](Use_Your_Own_SAE.ipynb) | Custom checkpoints and model adapter API | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenSAE/SAE-Behavioral-Compositionality/blob/main/notebooks/Use_Your_Own_SAE.ipynb) |

**In Colab:** click the badge and choose **Runtime → Run all**. No Hugging Face token is needed for the public GPT-2 SAE. The setup cell installs the reference `transformer-lens==3.5.0` and `sae-lens==6.44.4` versions; Colab provides PyTorch. GPU is optional but helps.

**From a local checkout:** open either notebook with Jupyter from the repository root, after installing the existing `requirements.txt`. The setup cell detects local `sar.py` and `sar_runner.py` without cloning.

**To connect a new checkpoint:** in `Use_Your_Own_SAE.ipynb`, set `MY_SAE` to an object with `.encode(torch.Tensor)` and row-major `.W_dec`, and set the matching `HOOK_NAME` and model. For arbitrary models, implement `FunctionalModelAdapter` and `FunctionalSAE` as the second example demonstrates. Both invoke the unchanged SAR estimator.

**Limitations:** Notebook execution with actual public pretrained weights depends on external downloads and compatible GPU/CPU runtime. Unit tests validate the adapter math and a fully simulated intervention backend; they do not substitute for an online Colab smoke test. For full experimental reproduction, see the source paper; these notebooks are teaching and integration examples.

**Colab URLs become live after these new notebook files are committed to the public repository.**
