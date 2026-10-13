"""SAR demo: GPT-2 Small + a public SAE, 24 copying prompts, K = 16 features, 384 coalitions per prompt.

python demo/demo.py                       # layers 4 and 10
python demo/demo.py --layers 4 6 8 10     # all paper layers for GPT-2 Small
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from sar import coalitions, design, rotation, sar

warnings.filterwarnings("ignore", category=UserWarning)
# (A, B, C): the prompt "A B A" should continue with B. The behavior is g = logit(B) - logit(C).
PROMPTS = [("Alice", "Emma", "Rachel"), ("Peter", "Steven", "Nina"), ("Julia", "Andrew", "Mary"),
           ("Nina", "George", "Anna"), ("Laura", "Helen", "John"), ("John", "Ruth", "Mary"),
           ("Anna", "Kate", "Oliver"), ("Sophie", "Peter", "Robert"), ("Diana", "Andrew", "Lucy"),
           ("Paul", "Sophie", "Steven"), ("Maria", "Michael", "Mark"), ("Claire", "Daniel", "Simon"),
           ("Martin", "Simon", "James"), ("Mark", "Grace", "Diana"), ("Ruth", "Martin", "Anna"),
           ("Kate", "Oliver", "George"), ("David", "Maria", "Simon"), ("Emma", "James", "John"),
           ("Michael", "George", "Peter"), ("Daniel", "Kate", "Sophie"), ("Simon", "Steven", "John"),
           ("Julia", "Thomas", "Andrew"), ("John", "Victor", "Anna"), ("Diana", "Alice", "Victor")]
K, ROTATIONS = 16, 8
HAT = r"$\widehat{\mathrm{SAR}}_1$"


def top_contributions(model, sae, hook, tokens):
    """Residual stream at the SAE site, h at the last token, top-K features, and their contributions z_i d_i."""
    _, cache = model.run_with_cache(tokens, names_filter=hook)
    resid = cache[hook]
    h = resid[0, -1]
    z = sae.encode(h[None])[0]
    top = torch.argsort(-z.abs(), stable=True)[:K]
    return resid, h, top, z[top, None] * sae.W_dec[top]


def behavior(model, layer, resid, h_tilde, ids):
    """Put each h_tilde at the last token, run the upper layers, and read logit(B) - logit(C)."""
    x = resid.repeat(len(h_tilde), 1, 1)
    x[:, -1] = h_tilde
    out = model(x, start_at_layer=layer, stop_at_layer=model.cfg.n_layers)[:, -1:]
    logits = model.ln_final(out)[:, 0] @ model.W_U[:, ids] + model.b_U[ids]
    return (logits[:, 0] - logits[:, 1]).double().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 10])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2", device=device)

    res = {}
    for layer in args.layers:
        hook = f"blocks.{layer}.hook_resid_pre"
        sae = SAE.from_pretrained("gpt2-small-res-jb", hook, device=device)
        sae = sae[0] if isinstance(sae, tuple) else sae
        rows = []
        for n, (a, b, c) in enumerate(PROMPTS):
            tokens = model.to_tokens(f"{a} {b} {a}")
            ids = [model.to_single_token(" " + b), model.to_single_token(" " + c)]
            resid, h, top, delta = top_contributions(model, sae, hook, tokens)
            masks, fit, held = coalitions(K, seed=n)             # same coalitions at every layer and rotation
            drop = torch.tensor(~masks, dtype=torch.float32, device=device)

            # h_tilde(S) = h - sum of the contributions left out of S
            g = behavior(model, layer, resid, h - drop @ delta, ids)
            g_full = behavior(model, layer, resid, h[None], ids)[0]
            out = sar(g, masks, fit, held, g_full=g_full)
            # Affine control: psi(h) = W_U[:, B] . h gives SAR_1 = 0 up to rounding.
            w = model.W_U[:, ids[0]].double()
            out["affine"] = sar(((h.double() - drop.double() @ delta.double()) @ w).cpu().numpy(),
                                masks, fit, held)["sar1"]
            # Same subspace, same total contribution, rotated coordinates: delta_Q = Q^T delta.
            out["rot"] = np.median([sar(behavior(model, layer, resid, h - drop @ (Q.T @ delta), ids),
                                        masks, fit, held)["sar1"]
                                    for Q in (torch.tensor(rotation(K, seed=1000 + ROTATIONS * n + j),
                                                           dtype=torch.float32, device=device)
                                              for j in range(ROTATIONS))])
            out["top"], out["g"], out["masks"], out["fit"] = top.tolist(), g, masks, fit
            rows.append(out)
        res[layer] = rows
        print(f"layer {layer:2d} done", flush=True)

    # 1. First order residual, SAE coordinates vs rotations, pairwise share (Figures 2, 3, 6).
    print(f"\nGPT-2 Small, {len(PROMPTS)} copying prompts, K = {K}, 384 coalitions (256 fit, 128 held out)\n")
    print(f"{'Layer':>5s} {'SAR_1 SAE':>10s} {'SAR_1 rotated':>14s} {'SAE lower':>10s} {'rho_2':>7s} "
          f"{'affine SAR_1':>13s}")
    for layer, rows in res.items():
        col = lambda k: np.array([r[k] for r in rows])
        print(f"{layer:5d} {np.median(col('sar1')):10.4f} {np.median(col('rot')):14.4f} "
              f"{int(np.sum(col('sar1') < col('rot'))):6d} / {len(rows)} {np.nanmedian(col('rho2')):7.3f} "
              f"{col('affine').max():13.1e}")

    # 2. First order vs pairwise explanations on held-out coalitions (Figure 5).
    print(f"\n{'':5s}  {'Rank fidelity':^17s}  {'Ordering error':^17s}  {'Sign error':^17s}  "
          f"{'Selection regret':^17s}")
    print(f"{'Layer':>5s}" + f"  {'first':>8s} {'pairwise':>8s}" * 4)
    for layer, rows in res.items():
        m = lambda k: np.mean([r[k] for r in rows])
        print(f"{layer:5d}" + "".join(f"  {m(k + '1'):8.4f} {m(k + '2'):8.4f}" for k in ("rank", "E", "sign", "regret")))

    # 3. Example: the most strongly interacting feature pair of the first prompt at the first layer.
    layer, r = args.layers[0], res[args.layers[0]][0]
    theta = np.linalg.lstsq(design(r["masks"], 2)[r["fit"]], r["g"][r["fit"]], rcond=None)[0]
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    p = int(np.argmax(np.abs(theta[1 + K:])))
    (i, j), (a, b, _) = pairs[p], PROMPTS[0]
    print(f'\nExample: "{a} {b} {a}", layer {layer}: SAR_1 = {r["sar1"]:.4f}, rho_2 = {r["rho2"]:.3f}')
    print(f"  strongest pair: features {r['top'][i]} and {r['top'][j]}, interaction {theta[1 + K + p]:+.3f}, "
          f"main effects {theta[1 + i]:+.3f} and {theta[1 + j]:+.3f}")
    for f in (r["top"][i], r["top"][j]):
        print(f"  https://www.neuronpedia.org/gpt2-small/{layer}-res-jb/{f}")

    layers, sae_c, base_c = list(res), "#2a78d6", "#eb6834"
    med = lambda layer, k: np.nanmedian([r[k] for r in res[layer]])
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    # A. Per prompt: SAE coordinates vs rotated coordinates (Figure 3). Light to dark = early to late layer.
    ramp = ["#6da7ec", "#3987e5", "#1c5cab", "#104281"]
    for k, layer in enumerate(layers):
        rows = res[layer]
        ax[0].scatter([r["rot"] for r in rows], [r["sar1"] for r in rows], s=28, edgecolor="white", lw=0.6,
                      color=ramp[round(k * 3 / max(len(layers) - 1, 1))], label=f"layer {layer}")
    lim = [min(ax[0].get_xlim()[0], ax[0].get_ylim()[0]), max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])]
    ax[0].plot(lim, lim, color="#8f8e88", lw=1, ls="--")
    ax[0].text(lim[1], lim[1], "equal ", ha="right", va="bottom", color="#52514e", fontsize=8)
    ax[0].set(xscale="log", yscale="log", xlabel=HAT + " in rotated coordinates, median of 8",
              ylabel=HAT + " in SAE coordinates", title="A. SAE and rotated coordinates")
    ax[0].legend(frameon=False, fontsize=8)
    # B, C. Depth profiles (Figure 6).
    for key, lab, cl, ls in (("sar1", "SAE", sae_c, "-"), ("rot", "rotated", base_c, "--")):
        ax[1].plot(layers, [med(l, key) for l in layers], marker="o", ms=8, lw=2, ls=ls, color=cl, label=lab)
    ax[1].set(yscale="log", xlabel="Layer", ylabel="median " + HAT, xticks=layers, title="B. First order residual")
    lo, hi = ax[1].get_ylim()
    ticks = [t for t in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1) if lo <= t <= hi]
    ax[1].set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax[1].set_ylim(lo, hi)
    ax[1].minorticks_off()
    ax[1].legend(frameon=False, fontsize=8)
    ax[2].plot(layers, [med(l, "rho2") for l in layers], marker="o", ms=8, lw=2, color=sae_c)
    ax[2].set(ylim=(0, 1), xlabel="Layer", ylabel=r"median $\widehat{\rho}_2$", xticks=layers, title="C. Pairwise share")
    # D. Held-out rank fidelity of the first order and pairwise fits (Figure 5).
    for k, layer in enumerate(layers):
        y1, y2 = (np.mean([r[key] for r in res[layer]]) for key in ("rank1", "rank2"))
        ax[3].plot([k, k], [y1, y2], color="#c3c2b7", lw=2, zorder=1)
        ax[3].scatter([k], [y1], s=64, color=base_c, zorder=2, label="first order" if k == 0 else None)
        ax[3].scatter([k], [y2], s=64, color=sae_c, zorder=2, label="pairwise" if k == 0 else None)
    ax[3].set(xticks=range(len(layers)), xticklabels=[f"layer {l}" for l in layers], xlim=(-0.6, len(layers) - 0.4),
              ylabel="mean held out rank fidelity", title="D. First order and pairwise")
    ax[3].legend(frameon=False, fontsize=8, loc="lower right")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle(f"GPT-2 Small, {len(PROMPTS)} copying prompts, K = {K}")
    fig.tight_layout()
    name = "demo_layers" + "_".join(map(str, layers)) + ".png"
    fig.savefig(name, dpi=150)
    print(f"\nSaved {name}")


if __name__ == "__main__":
    main()
