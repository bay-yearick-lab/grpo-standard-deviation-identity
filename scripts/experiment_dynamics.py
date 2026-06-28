"""Controlled GRPO training dynamics: the identity's predictions, measured.

A real (stochastic) multi-prompt GRPO training loop, run on CPU, that tests the
closed forms of the paper as *dynamic* predictions rather than static accounting.
Each prompt is a one-dimensional Bernoulli-logit policy p_x = sigmoid(theta_x);
the population's initial difficulty is the real Big-Math solve-rate distribution.
At each step every prompt draws a group of G rollouts, the advantage is formed
(GRPO divides by the group std, Dr.GRPO/REINFORCE does not, DAPO resamples
degenerate groups), and theta_x is updated by the actual sampled gradient.

Three predictions are checked against what the run actually does:
  (A) silent groups   -- measured unanimous-group fraction vs  E[p^G+(1-p)^G];
  (B) reweighting     -- measured gradient mass by difficulty vs sqrt(p(1-p))
                          (GRPO) and p(1-p) (Dr.GRPO);
  (C) difficulty bias -- the initially-hardest prompts rise faster under GRPO
                          (and faster still under DAPO) than under Dr.GRPO.

Outputs: paper/figures/fig_experiment.pdf and data/experiment_dynamics.json.
Run: uv run python scripts/experiment_dynamics.py
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle as fs
import grpo_diagnostics as gd

fs.use_style()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGOUT = os.path.join(HERE, "paper", "figures")
DATAOUT = os.path.join(HERE, "data")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    return np.log(p / (1.0 - p))


def initial_difficulty(M, rng):
    """Subsample M interior Big-Math solve rates as the starting population."""
    try:
        import pandas as pd
        path = os.path.join(HERE, "data", "raw", "all",
                            "train-00000-of-00001.parquet")
        p = pd.read_parquet(path, columns=["llama8b_solve_rate"])[
            "llama8b_solve_rate"].to_numpy(dtype=float)
        p = p[(p > 0) & (p < 1)]
        idx = rng.choice(len(p), size=M, replace=len(p) < M)
        return np.clip(p[idx], 0.02, 0.98), "Big-Math solve rates"
    except Exception:
        a = rng.beta(0.7, 1.6, M)
        b = rng.beta(1.6, 0.7, M)
        pick = rng.random(M) < 0.5
        return np.clip(np.where(pick, a, b), 0.02, 0.98), "Beta mixture (fallback)"


def draw_group(p, G, rng):
    """k correct out of G for every prompt simultaneously."""
    return (rng.random((len(p), G)) < p[:, None]).sum(1)


def grpo_gain(k, G):
    """GRPO per-prompt scalar gradient gain = sigma = sqrt(k(G-k))/G."""
    return np.sqrt(k * (G - k)) / G


def drgrpo_gain(k, G):
    """Dr.GRPO/REINFORCE per-prompt scalar gradient gain = sigma^2 = k(G-k)/G^2."""
    return k * (G - k) / (G * G)


def run(arm, p0, G, lr, T, rng, track_mask, dapo_tries=4):
    theta = logit(p0.copy())
    M = len(p0)
    silent_meas, silent_pred, hard_curve = [], [], []
    samples_used = 0
    for t in range(T):
        p = sigmoid(theta)
        hard_curve.append(float(p[track_mask].mean()))
        silent_pred.append(float(np.mean(gd.silent_rate(p, G))))
        k = draw_group(p, G, rng)
        samples_used += M * G
        degenerate = (k == 0) | (k == G)
        silent_meas.append(float(np.mean(degenerate)))
        if arm == "dapo":
            bad = degenerate.copy()
            tries = 1
            while bad.any() and tries < dapo_tries:
                k[bad] = draw_group(p[bad], G, rng)
                samples_used += int(bad.sum()) * G
                bad = (k == 0) | (k == G)
                tries += 1
            degenerate = (k == 0) | (k == G)
        gain = grpo_gain(k, G) if arm in ("grpo", "dapo") else drgrpo_gain(k, G)
        gain[degenerate] = 0.0
        theta = theta + lr * gain
    p = sigmoid(theta)
    hard_curve.append(float(p[track_mask].mean()))
    return {
        "p": p,
        "silent_meas": np.array(silent_meas),
        "silent_pred": np.array(silent_pred),
        "hard_curve": np.array(hard_curve),
        "samples_used": int(samples_used),
    }


def measure_reweighting(p0, G, rng, repeats=400):
    """Measured gradient mass by difficulty bin, GRPO vs Dr.GRPO, on the start pop.

    The overlaid prediction is the *finite-G* exact expected gradient per prompt
    (GRPO: E[sigma]; Dr.GRPO: p(1-p)(1-1/G)), not the large-G sqrt(p(1-p)) weight;
    at G=8 the two differ near the extremes by exactly the attenuation, and the
    measured mass follows the finite-G form. The large-G shares are returned too.
    """
    edges = np.linspace(0, 1, 11)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binidx = np.clip(np.digitize(p0, edges) - 1, 0, 9)
    mg = np.zeros(10)
    md = np.zeros(10)
    for _ in range(repeats):
        k = draw_group(p0, G, rng)
        gg, dd = grpo_gain(k, G), drgrpo_gain(k, G)
        for b in range(10):
            m = binidx == b
            mg[b] += gg[m].sum()
            md[b] += dd[m].sum()
    mg /= mg.sum()
    md /= md.sum()

    def binshare(w):
        s = np.array([w[binidx == b].sum() for b in range(10)])
        return s / s.sum()

    pg = binshare(gd.expected_gradient(p0, G))           # finite-G GRPO
    pr = binshare(p0 * (1 - p0) * (1 - 1 / G))            # finite-G Dr.GRPO
    lg = binshare(gd.mass_weight(p0, True))              # large-G GRPO sqrt
    lr_ = binshare(gd.mass_weight(p0, False))            # large-G REINFORCE
    return centers, mg, md, pg, pr, lg, lr_


def main():
    rng = np.random.default_rng(7)
    M, G, lr, T = 6000, 8, 0.5, 150
    p0, src = initial_difficulty(M, rng)
    hard = p0 <= np.quantile(p0, 0.25)

    res = {a: run(a, p0, G, lr, T, np.random.default_rng(100 + i), hard)
           for i, a in enumerate(["grpo", "drgrpo", "dapo"])}

    # (A) silent-rate fidelity, GRPO arm
    sm, sp = res["grpo"]["silent_meas"], res["grpo"]["silent_pred"]
    r2_silent = 1 - np.sum((sm - sp) ** 2) / np.sum((sm - sm.mean()) ** 2)
    max_abs = float(np.max(np.abs(sm - sp)))

    # (B) reweighting: measured vs finite-G closed form (and large-G reference)
    centers, mg, md, pg, pr, lg, lr_ = measure_reweighting(
        p0, G, np.random.default_rng(3))
    ext = (centers < 0.1) | (centers > 0.9)
    reweight = {
        "grpo_extreme_meas": float(mg[ext].sum()),
        "reinforce_extreme_meas": float(md[ext].sum()),
        "grpo_extreme_finiteG": float(pg[ext].sum()),
        "reinforce_extreme_finiteG": float(pr[ext].sum()),
        "grpo_extreme_largeG": float(lg[ext].sum()),
        "reinforce_extreme_largeG": float(lr_[ext].sum()),
    }

    out = {
        "source": src, "M": M, "G": G, "lr": lr, "T": T,
        "r2_silent": float(r2_silent), "silent_max_abs_err": max_abs,
        "reweight": reweight,
        "samples_grpo": res["grpo"]["samples_used"],
        "samples_dapo": res["dapo"]["samples_used"],
        "dapo_oversample_factor": res["dapo"]["samples_used"]
        / res["grpo"]["samples_used"],
        "hard_final_grpo": float(res["grpo"]["hard_curve"][-1]),
        "hard_final_drgrpo": float(res["drgrpo"]["hard_curve"][-1]),
        "hard_final_dapo": float(res["dapo"]["hard_curve"][-1]),
    }
    with open(os.path.join(DATAOUT, "experiment_dynamics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    # ----------------------------- figure -----------------------------
    steps = np.arange(T + 1)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(11.0, 3.5))
    fig.subplots_adjust(top=0.93, bottom=0.13, left=0.055, right=0.985,
                        wspace=0.33)

    axA.plot(steps[:T], sp, color=fs.NAVY, lw=1.8, zorder=3,
             label=r"closed form $\mathbb{E}[p^G{+}(1{-}p)^G]$")
    axA.scatter(steps[:T:8], sm[::8], s=15, color=fs.CLAY, zorder=4,
                label="measured in run")
    axA.set_xlabel("training step")
    axA.set_ylabel("silent-group fraction")
    axA.set_ylim(0, max(sm.max(), sp.max()) * 1.18)
    axA.set_title("(a)", fontsize=9.5, color=fs.AXIS, loc="left", pad=6)
    axA.legend(frameon=False, loc="upper right", handlelength=1.4, fontsize=7.6)
    fs.clean(axA)

    w = 0.45 * (centers[1] - centers[0])
    axB.bar(centers - w / 2, md, width=w, color=fs.CLAY, label="Dr. GRPO measured")
    axB.bar(centers + w / 2, mg, width=w, color=fs.NAVY, label="GRPO measured")
    axB.scatter(centers - w / 2, pr, s=22, color=fs.INK, zorder=5, marker="_",
                linewidths=1.4)
    axB.scatter(centers + w / 2, pg, s=22, color=fs.INK, zorder=5, marker="_",
                linewidths=1.4, label="closed form")
    axB.set_xlabel(r"difficulty $\hat p$ (start)")
    axB.set_ylabel("share of gradient mass")
    axB.set_xlim(0, 1)
    axB.set_ylim(0, 0.17)
    axB.set_title("(b)", fontsize=9.5, color=fs.AXIS, loc="left", pad=6)
    axB.legend(frameon=False, loc="upper center", ncol=3, handlelength=1.1,
               columnspacing=1.1, handletextpad=0.4, fontsize=7.0)
    fs.clean(axB)

    axC.plot(steps, res["drgrpo"]["hard_curve"], color=fs.CLAY, lw=1.8,
             label="Dr. GRPO")
    axC.plot(steps, res["grpo"]["hard_curve"], color=fs.NAVY, lw=1.8,
             label="GRPO")
    axC.plot(steps, res["dapo"]["hard_curve"], color=fs.GREY, lw=1.8, ls="--",
             label="DAPO")
    axC.set_xlabel("training step")
    axC.set_ylabel("mean solve rate, hardest quartile")
    axC.set_ylim(0, 1)
    axC.set_title("(c)", fontsize=9.5, color=fs.AXIS, loc="left", pad=6)
    axC.legend(frameon=False, loc="lower right", handlelength=1.6, fontsize=7.6)
    fs.clean(axC)

    fs.header(fig,
              "A controlled GRPO run confirms the identity's predictions step by step",
              f"{M:,} Bernoulli-logit prompts, initial difficulty from {src}; "
              f"group size $G={G}$.")
    fs.source(fig, "Controlled simulation; no LLM. scripts/experiment_dynamics.py.")
    fig.savefig(os.path.join(FIGOUT, "fig_experiment.pdf"))
    plt.close(fig)
    print("wrote fig_experiment.pdf")


if __name__ == "__main__":
    main()
