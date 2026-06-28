"""Theory exhibits for the paper (Exhibits 1 and 2)."""

import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs

fs.use_style()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "paper", "figures")
os.makedirs(OUT, exist_ok=True)


def grpo_advantage(r, normalize=True):
    mu = r.mean()
    if not normalize:
        return r - mu
    sd = r.std()
    return (r - mu) / sd if sd > 0 else np.zeros_like(r)


def expected_gradient(p, G, n, rng, normalize=True):
    tot = 0.0
    for _ in range(n):
        y = (rng.random(G) < p).astype(float)
        tot += np.mean(grpo_advantage(y, normalize) * (y - p))
    return tot / n


def exhibit_objective():
    rng = np.random.default_rng(0)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.5))
    fig.subplots_adjust(top=0.95, bottom=0.13, left=0.075, right=0.965,
                        wspace=0.30)

    # left: implicit objectives
    p = np.linspace(1e-4, 1 - 1e-4, 600)
    axL.plot(p, p, color=fs.CLAY, lw=2.2)
    axL.plot(p, (2 / np.pi) * np.arcsin(np.sqrt(p)), color=fs.NAVY, lw=2.2)
    axL.plot([0, 1], [0, 1], color=fs.GREY, lw=0.7, ls=(0, (2, 3)))
    axL.text(0.30, 0.78, "GRPO\n$(2/\\pi)\\,\\arcsin\\sqrt{p}$",
             color=fs.NAVY, fontsize=8.6, fontweight="bold", ha="left")
    axL.text(0.60, 0.40, "Dr. GRPO\n$p$", color=fs.CLAY, fontsize=8.6,
             fontweight="bold", ha="left")
    axL.set_xlabel("per-prompt success probability  $p$")
    axL.set_ylabel("implicit objective (normalized)")
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1)
    fs.clean(axL)

    # right: gradient verification
    pp = np.linspace(1e-3, 1 - 1e-3, 400)
    axR.plot(pp, np.sqrt(pp * (1 - pp)), color=fs.NAVY, lw=2.2, zorder=3)
    axR.plot(pp, pp * (1 - pp), color=fs.CLAY, lw=2.2, zorder=3)
    G = 64
    p_mc = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95])
    g_grpo = [expected_gradient(p_, G, 50_000, rng, True) for p_ in p_mc]
    g_rein = [expected_gradient(p_, G, 50_000, rng, False) for p_ in p_mc]
    axR.scatter(p_mc, g_grpo, s=26, facecolor="white", edgecolor=fs.NAVY,
                lw=1.3, zorder=5)
    axR.scatter(p_mc, g_rein, s=24, marker="s", facecolor="white",
                edgecolor=fs.CLAY, lw=1.3, zorder=5)
    axR.text(0.50, 0.575, "GRPO   $\\sqrt{p(1-p)}$", color=fs.NAVY,
             fontsize=8.6, fontweight="bold", ha="center")
    axR.text(0.50, 0.305, "Dr. GRPO   $p(1-p)$", color=fs.CLAY,
             fontsize=8.6, fontweight="bold", ha="center")
    axR.set_xlabel("per-prompt success probability  $p$")
    axR.set_ylabel(r"expected gradient in $\theta$")
    axR.set_xlim(0, 1)
    axR.set_ylim(0, 0.66)
    fs.clean(axR)

    fs.header(fig,
              "Averaged over groups, the identity is the arcsine gradient",
              "The group-standard-deviation identity averages to $\\sqrt{p(1-p)}$, the "
              "gradient of the variance-stabilized objective $\\arcsin\\sqrt{p}$.")
    fs.source(fig, "Markers: Monte Carlo over Bernoulli-logit prompts, "
                   "$G=64$. Lines: closed forms (large-group limit).")
    fig.savefig(os.path.join(OUT, "fig_implicit_objective.pdf"))
    plt.close(fig)
    print("wrote fig_implicit_objective.pdf")


def exhibit_identity():
    """Hero exhibit: the group-standard-deviation identity g(k)=sqrt(k(G-k))/G,
    baseline-free, and its finite-G attenuation toward the large-group limit."""
    rng = np.random.default_rng(2)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.5))
    fig.subplots_adjust(top=0.95, bottom=0.13, left=0.085, right=0.965,
                        wspace=0.30)

    # left: the identity g(k) = sqrt(k(G-k))/G, with Monte-Carlo markers at
    # three different baselines b to show baseline-independence.
    G = 8
    ks = np.arange(0, G + 1)
    closed = np.sqrt(ks * (G - ks)) / G
    axL.plot(ks, closed, color=fs.NAVY, lw=2.2, zorder=3)
    baselines = [(0.2, "o", fs.CLAY), (0.5, "s", fs.NAVY), (0.8, "^", fs.GREY)]
    mk_handles = []
    for b, mk, c in baselines:
        gk = []
        for k in ks:
            r = np.array([1.0] * k + [0.0] * (G - k))
            a = grpo_advantage(r, True)
            gk.append(float(np.mean(a * (r - b))))  # score with baseline b
        axL.scatter(ks, gk, s=24, marker=mk, facecolor="white", edgecolor=c,
                    lw=1.2, zorder=5)
        mk_handles.append(Line2D([0], [0], marker=mk, linestyle="none",
                                 markerfacecolor="white", markeredgecolor=c,
                                 markeredgewidth=1.2, markersize=5,
                                 label=rf"$b={b}$"))
    axL.text(4.0, 0.62, r"$g(k)=\dfrac{\sqrt{k(G{-}k)}}{G}=\sigma$",
             color=fs.NAVY, fontsize=9.2, fontweight="bold", ha="center",
             va="center")
    axL.text(4.0, 0.165, "Monte Carlo, baseline $b$ (coincide):",
             color=fs.AXIS, fontsize=7.4, ha="center", va="center")
    axL.legend(handles=mk_handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.8, handletextpad=0.3, columnspacing=1.2,
               borderaxespad=0.9)
    axL.set_xlabel(r"correct samples in the group  $k$  (of $G=8$)")
    axL.set_ylabel(r"per-prompt gradient  $g(k)$")
    axL.set_xlim(-0.3, G + 0.3)
    axL.set_ylim(0, 0.70)
    axL.set_xticks(ks)
    fs.clean(axL)

    # right: finite-G attenuation toward the large-group limit, with the
    # delta-method curve 1 - 1/(8 G p(1-p)) overlaid on Monte Carlo.
    Gs = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256])
    Gline = np.linspace(4, 256, 400)
    for p0, c, mk in [(0.5, fs.NAVY, "o"), (0.2, fs.CLAY, "s")]:
        att = [expected_gradient(p0, int(Gv), 80_000, rng, True) /
               np.sqrt(p0 * (1 - p0)) for Gv in Gs]
        axR.plot(Gs, att, color=c, lw=0, marker=mk, ms=4.5,
                 markerfacecolor="white", markeredgewidth=1.2, zorder=5)
        axR.plot(Gline, 1 - 1 / (8 * Gline * p0 * (1 - p0)), color=c,
                 lw=1.6, zorder=3)
    axR.axhline(1.0, color=fs.GREY, lw=0.8, ls=(0, (2, 3)))
    axR.text(150, 0.93, "$p=0.5$", color=fs.NAVY, fontsize=8.4, fontweight="bold")
    axR.text(150, 0.81, "$p=0.2$", color=fs.CLAY, fontsize=8.4, fontweight="bold")
    axR.set_xscale("log")
    axR.set_xlabel("group size  $G$")
    axR.set_ylabel(r"gradient realized  $\mathbb{E}[g]/\sqrt{p(1-p)}$")
    axR.set_xlim(3.5, 300)
    axR.set_ylim(0.4, 1.05)
    axR.set_xticks([4, 8, 16, 32, 64, 128, 256])
    axR.set_xticklabels([4, 8, 16, 32, 64, 128, 256])
    fs.clean(axR)

    fs.header(fig,
              "GRPO grades a prompt by how much its own samples disagreed",
              "Left: the gradient equals the group reward std, the same for any "
              "baseline. Right: the delta-method law $1-1/(8Gp(1-p))$ tracks it.")
    fs.source(fig, "Markers: Monte Carlo, Bernoulli-logit prompts. "
                   "Lines: closed forms (Proposition 1).")
    fig.savefig(os.path.join(OUT, "fig_identity.pdf"))
    plt.close(fig)
    print("wrote fig_identity.pdf")


def exhibit_finite_group():
    """Silent-group rate p^G + (1-p)^G across group sizes."""
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 3.5))
    fig.subplots_adjust(top=0.95, bottom=0.13, left=0.085, right=0.965)

    p = np.linspace(1e-3, 1 - 1e-3, 600)
    shades = {4: "#9DB6CE", 8: "#5E86AD", 16: "#345D87", 64: fs.NAVY}
    for G, c in shades.items():
        ax.plot(p, p**G + (1 - p)**G, color=c, lw=2.2, label=f"$G={G}$")
    ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=8.6,
              columnspacing=2.0, handlelength=1.5, handletextpad=0.5,
              bbox_to_anchor=(0.5, 1.0))
    ax.set_xlabel(r"per-prompt success probability  $p$")
    ax.set_ylabel(r"silent-group rate  $p^{G}+(1-p)^{G}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fs.clean(ax)

    fs.header(fig,
              "Small groups silence easy and hard prompts",
              "The silent-group rate $p^{G}+(1-p)^{G}$: the $\\sigma=0$ mass that "
              "dynamic sampling discards, geometric in $G$ but pinned at $p\\in\\{0,1\\}$.")
    fs.source(fig, "Closed form: the silent-group rate (Section on dynamic sampling).")
    fig.savefig(os.path.join(OUT, "fig_finite_group.pdf"))
    plt.close(fig)
    print("wrote fig_finite_group.pdf")


def exhibit_difficulty_weighting():
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 3.5))
    fig.subplots_adjust(top=0.95, bottom=0.13, left=0.085, right=0.965)

    # The difficulty-bias weight: w(p) = d/dp [2 arcsin sqrt(p)] = 1/sqrt(p(1-p)),
    # the same 1/sigma that appears in the GRPO advantage. REINFORCE / Dr.GRPO
    # place a flat weight w == 1 (they ascend the raw success rate p).
    p = np.linspace(1e-3, 1 - 1e-3, 800)
    w = 1.0 / np.sqrt(p * (1 - p))
    ax.plot(p, w, color=fs.NAVY, lw=2.2, zorder=3)
    ax.axhline(1.0, color=fs.CLAY, lw=2.2, zorder=3)

    # minimum of the bathtub: w(1/2) = 2 (label set clear, to the upper right
    # of the marker, in the gap between the GRPO curve and the flat baseline)
    ax.scatter([0.5], [2.0], s=26, facecolor="white", edgecolor=fs.NAVY,
               lw=1.3, zorder=5)
    ax.annotate(r"$w(1/2)=2$", xy=(0.5, 2.0), xytext=(0.605, 2.85),
                color=fs.NAVY, fontsize=8.4, fontweight="bold", ha="left",
                va="center",
                arrowprops=dict(arrowstyle="-", color=fs.NAVY, lw=0.8,
                                shrinkA=1.5, shrinkB=3.0))
    ax.text(0.5, 1.0, "  Dr. GRPO   $w\\equiv1$",
            color=fs.CLAY, fontsize=8.4, fontweight="bold", ha="center",
            va="bottom")
    ax.text(0.30, 4.6, "GRPO   $w(p)=1/\\sqrt{p(1-p)}$",
            color=fs.NAVY, fontsize=8.8, fontweight="bold", ha="left")

    ax.set_yscale("log")
    ax.set_xlabel("per-prompt success probability  $p$")
    ax.set_ylabel(r"gradient weight per unit $\Delta p$  (log scale)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.9, 40)
    fs.clean(ax)

    fs.header(fig,
              "The difficulty bias is the derivative of the arcsine transform",
              "GRPO weights a unit of success probability by $1/\\sqrt{p(1-p)}$, "
              "diverging at easy and hard prompts; Dr. GRPO keeps it flat.")
    fs.source(fig, "Closed form $w(p)=\\partial_p\\,2\\arcsin\\sqrt{p}$ "
                   "(large-group limit of the identity).")
    fig.savefig(os.path.join(OUT, "fig_difficulty_weighting.pdf"))
    plt.close(fig)
    print("wrote fig_difficulty_weighting.pdf")


def _load_dapo_curve():
    """Digitized DAPO Fig. 3b: discarded-prompt (accuracy=1) fraction vs step."""
    path = os.path.join(HERE, "data", "dapo",
                        "dapo_fig3b_accuracy1_fraction.csv")
    steps, frac = [], []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or row[0] == "step":
                continue
            steps.append(float(row[0]))
            frac.append(float(row[1]))
    return np.array(steps), np.array(frac)


def _bigmath_solve_rates():
    import pandas as pd
    path = os.path.join(HERE, "data", "raw", "all",
                        "train-00000-of-00001.parquet")
    return pd.read_parquet(path, columns=["llama8b_solve_rate"])[
        "llama8b_solve_rate"].to_numpy(dtype=float)


def exhibit_dapo():
    """Corroborate the silent-group identification against DAPO's real run.

    The structural claim is exact and unfitted: DAPO's dynamic sampling discards
    groups with accuracy 0 or 1, which is the silent mass p^G+(1-p)^G of the
    identity; its logged "accuracy = 1 fraction" (Fig. 3b, avg@32 = 100%) is the
    all-correct branch E[p^32] of that same mass. The TIME EVOLUTION is a
    consistency check, not a unique prediction: the all-correct mass on a
    sharpening Big-Math distribution reproduces the curve's shape with one free
    timescale (plateau pinned, shape fixed by Big-Math), and a generic 2-param
    saturating curve fits at least as well. Both R^2 values are reported so the
    comparison is transparent.

    Left: DAPO's logged curve with the closed-form fit, its bootstrap band, and
    the generic saturating baseline. Right: the closed-form functional form, the
    all-correct mass E[p^32] vs competence on Big-Math, with DAPO's plateau.
    """
    from scipy import optimize

    steps, frac = _load_dapo_curve()
    p = _bigmath_solve_rates()
    n, G = 32, 16  # n = 32 evaluation samples (avg@32); training group size 16
    plateau = frac[-8:].mean()
    s0 = steps.min()

    def warp(pr, T):
        return np.clip(pr ** (1.0 / T), 0, 1)

    def allcorr(T, m=n):
        return np.mean(warp(p, T) ** m)

    # competence -> mastery lookup; plateau pins T_end (no free plateau param)
    T_end = optimize.brentq(lambda T: allcorr(T) - plateau, 1.0, 2000.0)
    cgrid = np.linspace(0, 1, 400)
    acc = np.array([allcorr(1 + c * (T_end - 1)) for c in cgrid])

    def predict(tau, st=steps):
        c = 1.0 - np.exp(-(st - s0) / tau)
        c = (c - c.min()) / (c.max() - c.min())
        return np.interp(c, cgrid, acc)

    def r2_of(pr, fr=frac):
        return 1.0 - np.sum((pr - fr) ** 2) / np.sum((fr - fr.mean()) ** 2)

    res = optimize.minimize_scalar(
        lambda lt: np.sum((predict(np.exp(lt)) - frac) ** 2),
        bounds=(np.log(200), np.log(40000)), method="bounded")
    pred = predict(np.exp(res.x))
    r2 = r2_of(pred)

    # bootstrap band on the closed-form fit (resample points, refit timescale)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(300):
        idx = np.sort(rng.integers(0, len(steps), len(steps)))
        st, fr = steps[idx], frac[idx]

        def loss(lt):
            c = 1.0 - np.exp(-(st - s0) / np.exp(lt))
            c = (c - c.min()) / (c.max() - c.min() + 1e-9)
            return np.sum((np.interp(c, cgrid, acc) - fr) ** 2)

        rb = optimize.minimize_scalar(loss, bounds=(np.log(200), np.log(40000)),
                                      method="bounded")
        boot.append(predict(np.exp(rb.x)))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, 10, 0), np.percentile(boot, 90, 0)

    # generic 2-parameter saturating baseline (no theory): a(1 - e^{-(s-s0)/tau})
    rbase = optimize.least_squares(
        lambda par: par[0] * (1 - np.exp(-(steps - s0) / par[1])) - frac,
        [0.5, 2000.0], bounds=([0, 100], [1, 1e5]))
    base = rbase.x[0] * (1 - np.exp(-(steps - s0) / rbase.x[1]))
    r2_base = r2_of(base)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.5))
    fig.subplots_adjust(top=0.95, bottom=0.13, left=0.085, right=0.965,
                        wspace=0.30)

    # ---- left: logged curve, closed-form fit + band, generic baseline ----
    axL.fill_between(steps, lo, hi, color=fs.CLAY, alpha=0.16, lw=0, zorder=2)
    axL.plot(steps, base, color=fs.GREY, lw=1.3, ls=(0, (4, 2)), zorder=3,
             label=rf"generic saturating ($R^2{{=}}{r2_base:.2f}$)")
    axL.plot(steps, pred, color=fs.CLAY, lw=2.2, zorder=4,
             label=rf"closed form $\mathbb{{E}}[p^{{32}}]$ ($R^2{{=}}{r2:.2f}$)")
    axL.plot(steps, frac, color=fs.NAVY, lw=1.5, zorder=5,
             label="DAPO Fig. 3b, logged")
    axL.axhline(plateau, color=fs.GREY, lw=0.8, ls=(0, (2, 3)), zorder=1)
    axL.text(8600, plateau + 0.02, f"plateau {plateau:.2f}",
             color=fs.AXIS, fontsize=7.6, ha="right")
    axL.set_xlabel("training step")
    axL.set_ylabel("discarded-prompt fraction (acc $=1$)")
    axL.set_xlim(0, steps.max())
    axL.set_ylim(0, 0.66)
    axL.set_xticks([0, 2000, 4000, 6000, 8000])
    axL.legend(frameon=False, loc="lower right", fontsize=7.0,
               handlelength=1.5, labelspacing=0.3)
    fs.clean(axL)

    # ---- right: the functional form, all-correct mass vs competence ----
    Tgrid = np.geomspace(1.0, 400.0, 80)
    meanp = np.array([warp(p, T).mean() for T in Tgrid])
    law = np.array([allcorr(T) for T in Tgrid])
    axR.plot(meanp, law, color=fs.CLAY, lw=2.2, zorder=3)
    axR.scatter([p.mean()], [np.mean(p ** n)], s=30, marker="o",
                facecolor="white", edgecolor=fs.NAVY, lw=1.4, zorder=6)
    axR.text(p.mean() + 0.015, np.mean(p ** n) + 0.01,
             "Big-Math\n(frozen)", color=fs.NAVY, fontsize=7.6, ha="left",
             va="bottom")
    m_star = np.interp(plateau, law, meanp)
    axR.scatter([m_star], [plateau], s=34, marker="D", facecolor=fs.NAVY,
                edgecolor="white", lw=1.0, zorder=6)
    axR.text(m_star - 0.02, plateau, "DAPO\nplateau  ", color=fs.NAVY,
             fontsize=7.6, ha="right", va="center")
    axR.axhline(plateau, color=fs.GREY, lw=0.8, ls=(0, (2, 3)), zorder=2)
    axR.set_xlabel("policy competence  (mean solve rate)")
    axR.set_ylabel(r"all-correct mass  $\mathbb{E}[p^{32}]$  (avg@32)")
    axR.set_xlim(0.45, 0.97)
    axR.set_ylim(0, 0.85)
    fs.clean(axR)

    fs.header(fig,
              "DAPO discards exactly the silent mass; its logged curve is consistent",
              "DAPO's Fig. 3b (avg@32 $=100\\%$) is the all-correct branch "
              "$\\mathbb{E}[p^{32}]$ of the silent mass DAPO removes.")
    fs.source(fig, "Logged curve: DAPO (Yu et al., 2025), Fig. 3b, digitized. "
                   "Band: bootstrap 10-90%. Baseline: generic 2-parameter "
                   "saturating fit.")
    fig.savefig(os.path.join(OUT, "fig_dapo.pdf"))
    plt.close(fig)
    print(f"wrote fig_dapo.pdf  (plateau={plateau:.3f}, R^2_closed={r2:.3f}, "
          f"R^2_baseline={r2_base:.3f}, implied mean solve~{m_star:.2f})")


from math import lgamma


def fidelity(G, p):
    """Gradient fidelity phi(G,p) = E[g]/sqrt(p(1-p)), the fraction of the
    large-group (arcsine) gradient a group of size G realizes at difficulty p.
    Exact binomial sum, computed in log space so any G is safe."""
    ks = np.arange(1, G)
    logc = (lgamma(G + 1) - np.array([lgamma(k + 1) + lgamma(G - k + 1)
                                      for k in ks]))
    terms = np.exp(logc + ks * np.log(p) + (G - ks) * np.log1p(-p))
    Eg = (terms * np.sqrt(ks * (G - ks))).sum() / G
    return Eg / np.sqrt(p * (1 - p))


def required_G(p, target, Gmax=20000):
    """Smallest G with fidelity phi(G,p) >= target (exact)."""
    for G in range(2, Gmax):
        if fidelity(G, p) >= target:
            return G
    return None


def closed_G(eps, p):
    """The group-size law G* = 1/(8 eps p(1-p))."""
    return 1.0 / (8 * eps * p * (1 - p))


def exhibit_groupsize():
    """Showcase: the group-size law. Left: gradient fidelity climbing to one in
    G, the closed-form law tracking it. Right: the group size a target fidelity
    demands as a function of difficulty (the difficulty penalty), exact vs law."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.6))
    fig.subplots_adjust(top=0.95, bottom=0.13, left=0.082, right=0.965,
                        wspace=0.30)

    # left: fidelity vs group size for several difficulties
    Gs = np.unique(np.round(np.logspace(np.log10(2), np.log10(300), 70))).astype(int)
    shades = [(0.50, "#9BB6D2"), (0.10, "#3C6FA0"), (0.05, "#16314F")]
    for p0, c in shades:
        phi = np.array([fidelity(int(G), p0) for G in Gs])
        law = 1 - 1 / (8 * Gs * p0 * (1 - p0))
        axL.plot(Gs, phi, color=c, lw=2.2, zorder=4)
        axL.plot(Gs, law, color=c, lw=1.0, ls=(0, (2, 2)), alpha=0.85, zorder=3)
    axL.axhline(0.95, color=fs.GREY, lw=0.8, ls=(0, (1, 2)))
    axL.text(2.2, 0.955, "95%", color=fs.AXIS, fontsize=7.6, ha="left",
             va="bottom")
    for p0, c, y in [(0.50, "#9BB6D2", 0.965), (0.10, "#3C6FA0", 0.83),
                     (0.05, "#16314F", 0.58)]:
        axL.text(330, y, f"$p={p0:g}$", color=c, fontsize=8.6,
                 fontweight="bold", ha="left", va="center")
    axL.set_xscale("log")
    axL.set_xlabel("group size  $G$")
    axL.set_ylabel(r"gradient fidelity  $\varphi(G,p)$")
    axL.set_xlim(2, 470)
    axL.set_ylim(0.2, 1.02)
    axL.set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
    axL.set_xticklabels([2, 4, 8, 16, 32, 64, 128, 256])
    axL.legend([Line2D([0], [0], color=fs.NAVY, lw=2.1),
                Line2D([0], [0], color=fs.NAVY, lw=1.0, ls=(0, (2, 2)))],
               ["exact", r"law"], loc="lower right", frameon=False,
               fontsize=8.0, handlelength=1.8, borderaxespad=0.6)
    fs.clean(axL)

    # right: group size a target fidelity demands, vs difficulty
    ps = np.linspace(0.04, 0.96, 70)
    for eps, c in [(0.05, fs.NAVY), (0.01, fs.CLAY)]:
        exact = np.array([required_G(p, 1 - eps) for p in ps])
        law = closed_G(eps, ps)
        axR.plot(ps, exact, color=c, lw=2.1, zorder=4)
        axR.plot(ps, law, color=c, lw=1.1, ls=(0, (2, 2)), alpha=0.85, zorder=3)
    axR.text(0.5, 38, "99% fidelity", color=fs.CLAY, fontsize=8.8,
             fontweight="bold", ha="center", va="center")
    axR.text(0.5, 6.6, "95% fidelity", color=fs.NAVY, fontsize=8.8,
             fontweight="bold", ha="center", va="center")
    axR.set_yscale("log")
    axR.set_xlabel("per-prompt success probability  $p$")
    axR.set_ylabel(r"group size for target fidelity")
    axR.set_xlim(0, 1)
    axR.set_ylim(4, 400)
    axR.set_yticks([4, 8, 16, 32, 64, 128, 256])
    axR.set_yticklabels([4, 8, 16, 32, 64, 128, 256])
    fs.clean(axR)

    fs.header(fig,
              "Set the group size from difficulty, not by sweeping",
              "Gradient fidelity $\\varphi=\\mathbb{E}[g]/\\sqrt{p(1-p)}$, and "
              "the group size a target demands: harder prompts cost more.")
    fs.source(fig, "Closed forms: gradient fidelity and the group-size law "
                   "(Corollary 3). Exact: binomial sum.")
    fig.savefig(os.path.join(OUT, "fig_groupsize.pdf"))
    plt.close(fig)
    print("wrote fig_groupsize.pdf")


def write_groupsize_table():
    """Reproducible group-size-law table: exact required G (closed-form G* in
    parentheses), by difficulty and target fidelity."""
    tabdir = os.path.join(HERE, "paper", "tables")
    os.makedirs(tabdir, exist_ok=True)
    rows = [("$0.50$", 0.50), ("$0.30$ / $0.70$", 0.30),
            ("$0.10$ / $0.90$", 0.10), ("$0.05$ / $0.95$", 0.05)]
    epss = [(0.10, "$90\\%$"), (0.05, "$95\\%$"), (0.01, "$99\\%$")]
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r" & \multicolumn{3}{c}{group size $G$ for fidelity "
             r"$\varphi\ge 1-\varepsilon$} \\",
             r"\cmidrule(lr){2-4}",
             "difficulty $p$ & " + " & ".join(h for _, h in epss) + r" \\",
             r"\midrule"]
    for label, p in rows:
        cells = []
        for eps, _ in epss:
            ge = required_G(p, 1 - eps)
            gc = int(round(closed_G(eps, p)))
            cells.append(f"{ge} ({gc})")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(tabdir, "generated_groupsize_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote generated_groupsize_table.tex")


if __name__ == "__main__":
    exhibit_identity()
    exhibit_finite_group()
    exhibit_groupsize()
    exhibit_objective()
    exhibit_difficulty_weighting()
    exhibit_dapo()
    write_groupsize_table()
