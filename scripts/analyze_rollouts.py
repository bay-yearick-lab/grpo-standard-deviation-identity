"""Real-data exhibit and tables (Exhibit 3, Tables 1-2).

Reads the per-problem empirical solve rate p_hat = k/64 of Llama-3.1-8B on the
Big-Math corpus (open-r1/Big-Math-RL-Verified-Processed) and computes, on a
realistic difficulty distribution: the histogram of p_hat; the share of the
per-prompt gradient budget that GRPO (weight sqrt(p(1-p))) and REINFORCE
(weight p(1-p)) spend at each difficulty; and the fraction of prompts whose
group is unanimous (zero signal) as a function of group size G.
"""

import os
from math import comb

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle as fs
from huggingface_hub import hf_hub_download

fs.use_style()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGOUT = os.path.join(HERE, "paper", "figures")
TABOUT = os.path.join(HERE, "paper", "tables")
os.makedirs(FIGOUT, exist_ok=True)
os.makedirs(TABOUT, exist_ok=True)


def load_solve_rates():
    path = hf_hub_download(
        "open-r1/Big-Math-RL-Verified-Processed",
        "all/train-00000-of-00001.parquet",
        repo_type="dataset",
        local_dir=os.path.join(HERE, "data", "raw"),
    )
    return pd.read_parquet(path, columns=["llama8b_solve_rate"])[
        "llama8b_solve_rate"].to_numpy(dtype=float)


def main():
    p = load_solve_rates()
    N = len(p)
    frac0, frac1 = np.mean(p == 0), np.mean(p == 1)
    wR, wG = p * (1 - p), np.sqrt(p * (1 - p))

    ext = ((p > 0) & (p < 0.1)) | ((p > 0.9) & (p < 1))
    mid = (p >= 0.4) & (p <= 0.6)
    sR_ext, sG_ext = wR[ext].sum() / wR.sum(), wG[ext].sum() / wG.sum()
    sR_mid, sG_mid = wR[mid].sum() / wR.sum(), wG[mid].sum() / wG.sum()
    Gs = [4, 8, 16, 32, 64]
    zero = {G: np.mean(p**G + (1 - p)**G) for G in Gs}

    # Measured silent rate: draw size-G groups WITHOUT replacement directly from
    # the n=64 logged rollouts per problem (exact hypergeometric), no Bernoulli
    # assumption. Valid where G is well below the 64-rollout budget; the small
    # gap to the closed form is the finite-pool correction. (k=round(64*p_hat);
    # 6 of N problems sit on a /9 grid and round harmlessly.)
    n_roll = 64
    kk = np.round(p * n_roll).astype(int)
    counts = np.bincount(kk, minlength=n_roll + 1)
    Gs_meas = [4, 8, 16]

    def measured_silent(G):
        denom = float(comb(n_roll, G))
        tot = 0.0
        for c, cnt in enumerate(counts):
            if cnt == 0:
                continue
            allc = float(comb(c, G)) if c >= G else 0.0
            allw = float(comb(n_roll - c, G)) if (n_roll - c) >= G else 0.0
            tot += cnt * (allc + allw) / denom
        return tot / counts.sum()

    meas = {G: measured_silent(G) for G in Gs_meas}

    print(f"N={N}  unsolved={frac0:.3f}  trivial={frac1:.3f}")
    print(f"extreme mass  REINFORCE={sR_ext:.3f}  GRPO={sG_ext:.3f} "
          f"(x{sG_ext/sR_ext:.2f})")
    print(f"middle  mass  REINFORCE={sR_mid:.3f}  GRPO={sG_mid:.3f}")
    print("silent (closed form):", {G: round(z, 3) for G, z in zero.items()})
    print("silent (subsampled): ", {G: round(meas[G], 3) for G in Gs_meas})

    # ---------------- Exhibit 3 ----------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.5))
    fig.subplots_adjust(top=0.95, bottom=0.13, left=0.085, right=0.965,
                        wspace=0.30)

    axL.hist(p, bins=np.linspace(0, 1, 65), color=fs.NAVY, edgecolor="none")
    axL.set_xlabel(r"empirical solve rate  $\hat p$")
    axL.set_ylabel("number of problems")
    axL.set_xlim(0, 1)
    axL.annotate(f"{frac0*100:.0f}% never solved", xy=(0.0, frac0 * N),
                 xytext=(0.16, frac0 * N * 1.05), fontsize=8, color=fs.CLAY,
                 arrowprops=dict(arrowstyle="->", color=fs.CLAY, lw=0.9))
    axL.annotate(f"{frac1*100:.0f}% always solved", xy=(1.0, frac1 * N),
                 xytext=(0.42, frac1 * N * 0.86), fontsize=8, color=fs.CLAY,
                 arrowprops=dict(arrowstyle="->", color=fs.CLAY, lw=0.9))
    fs.clean(axL)

    edges = np.linspace(0, 1, 11)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shR, shG = [], []
    for i in range(10):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < 9 else \
            (p >= edges[i]) & (p <= edges[i + 1])
        shR.append(wR[m].sum() / wR.sum())
        shG.append(wG[m].sum() / wG.sum())
    w = 0.038
    axR.bar(centers - w, shR, width=2 * w, color=fs.CLAY, label="Dr. GRPO")
    axR.bar(centers + w, shG, width=2 * w, color=fs.NAVY, label="GRPO")
    axR.set_xlabel(r"problem difficulty (solve rate  $\hat p$)")
    axR.set_ylabel("share of total gradient budget")
    axR.set_xlim(0, 1)
    axR.set_ylim(0, 0.20)
    axR.legend(frameon=False, loc="upper center", ncol=2,
               handlelength=1.1, columnspacing=1.3)
    fs.clean(axR)

    fs.header(fig,
              "On real data, GRPO nearly doubles the effort spent on the extremes",
              f"Big-Math: $N={N:,}$ problems, Llama-3.1-8B solve rate over 64 rollouts.")
    fs.source(fig, "Source: Big-Math (Albalak et al., 2025), "
                   "open-r1/Big-Math-RL-Verified-Processed.")
    fig.savefig(os.path.join(FIGOUT, "fig_realdata.pdf"))
    plt.close(fig)
    print("wrote fig_realdata.pdf")

    # ---------------- tables ----------------
    mass = [
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"gradient budget on & Dr.\,GRPO & GRPO \\",
        r" & $p(1{-}p)$ & $\sqrt{p(1{-}p)}$ \\",
        r"\midrule",
        rf"extreme ($\hat p{{<}}.1$ or ${{>}}.9$) & {sR_ext*100:.1f}\% & {sG_ext*100:.1f}\% \\",
        rf"medium ($.4{{\le}}\hat p{{\le}}.6$) & {sR_mid*100:.1f}\% & {sG_mid*100:.1f}\% \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(os.path.join(TABOUT, "generated_realdata_mass_table.tex"), "w") as f:
        f.write("\n".join(mass) + "\n")

    meas_cell = {G: f"{meas[G]*100:.0f}\\%" for G in Gs_meas}
    deg = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"group size $G$ & 4 & 8 & 16 & 32 & 64 \\",
        r"\midrule",
        r"closed form & "
        + " & ".join(f"{zero[G]*100:.0f}\\%" for G in Gs) + r" \\",
        r"subsampled & "
        + " & ".join(meas_cell.get(G, "--") for G in Gs) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(os.path.join(TABOUT, "generated_realdata_degenerate_table.tex"), "w") as f:
        f.write("\n".join(deg) + "\n")
    print("wrote tables")


if __name__ == "__main__":
    main()
