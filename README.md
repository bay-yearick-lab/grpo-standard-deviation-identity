<p align="center">
  <h1 align="center">The Group-Standard-Deviation Identity</h1>
  <h3 align="center">GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number</h3>
  <p align="center">
    <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
    <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12-blue.svg">
  </p>
</p>

<p align="center">
  <strong>Yong Yi Bay</strong> &nbsp;&middot;&nbsp; <strong>Kathleen A. Yearick</strong><br>
  University of Illinois at Urbana-Champaign
</p>

---

A short note giving the exact finite-group identity behind GRPO's reward
standardization. The center is one identity, and it holds **in any dimension**:
for binary verifiable rewards with $k$ of $G$ samples correct, **the per-prompt
GRPO update equals the group's own empirical reward standard deviation
$\sigma=\sqrt{k(G-k)}/G$ times the contrastive direction between its correct and
incorrect rollouts**, $g=\sigma\,(\bar s_+ - \bar s_-)$, exactly and
independently of the baseline. The standard deviation is not a multiplier beside
the gradient; it *is* the gradient's magnitude. GRPO's signal for a prompt is
precisely how much its own samples disagreed: the **group-standard-deviation
identity**. Its group average is the arcsine variance-stabilizing gradient
understood in the large-group limit
([Thrampoulidis, Mahdavi & Deng 2025](https://arxiv.org/abs/2510.23049);
[Bartlett 1936](https://doi.org/10.2307/2983678),
[Anscombe 1948](https://doi.org/10.2307/2332343)), but the exact form governs a
real step.

From the identity follow, in closed form, the two knobs a practitioner sets:

- a **group-size law** $G\gtrsim 1/(8\varepsilon\,p(1-p))$ from the finite-$G$
  attenuation $1-1/(8Gp(1-p))$, for staying within a fraction $\varepsilon$ of
  the large-group gradient;
- the **silent-group rate** $p^G+(1-p)^G$, the $\sigma=0$ mass that DAPO's
  dynamic sampling discards. DAPO's logged accuracy-1 fraction is exactly the
  all-correct branch $\mathbb{E}[p^{32}]$ of this mass (a structural
  identification, not a fit); its time evolution is consistent with the closed
  form ($R^2=0.92$), though a generic saturating curve fits comparably, so the
  robust content is the identification and the sub-one plateau the geometric
  tail forces;
- the **difficulty bias** as the derivative
  $\partial_p\,2\arcsin\sqrt p = 1/\sqrt{p(1-p)}$ (deleted by Dr.GRPO), and
  group-mean centering as RLOO up to the constant $G/(G-1)$.

A **controlled GRPO run** (CPU, no LLM) confirms the closed forms as training
dynamics: the silent-group rate tracks the measured wasted-group fraction step
by step ($R^2=0.999$), the realized gradient mass matches the finite-$G$
reweighting, and GRPO lifts the hardest prompts where Dr. GRPO
stalls. Quantified on the **215,608-problem Big-Math** solve-rate corpus: the
standardization shifts the implicit objective's gradient mass onto difficulty
extremes ($13.9\\% \to 24.7\\%$) and leaves $44\\%$ of problems silent at $G=8$,
matching direct subsampling of the logged rollouts to within two points.

## Diagnostics (the formulas as a tiny API)

`scripts/grpo_diagnostics.py` exposes every closed form for use in a training
loop (`python scripts/grpo_diagnostics.py` prints a worked cheat sheet):

| quantity | closed form | function |
| --- | --- | --- |
| per-prompt gradient gain | `sqrt(k(G-k))/G = sigma` | `per_prompt_gradient(k, G)` |
| silent-group rate | `p**G + (1-p)**G` | `silent_rate(p, G)` |
| expected gradient (exact, finite G) | binomial sum | `expected_gradient(p, G)` |
| finite-G attenuation | `1 - 1/(8 G p(1-p))` | `attenuation(p, G)` |
| group-size budget | `1 / (8 eps p(1-p))` | `group_size_for_epsilon(eps, p)` |
| difficulty weight | `1 / sqrt(p(1-p))` | `difficulty_weight(p)` |

```python
import grpo_diagnostics as gd
gd.silent_rate(0.5, 8)              # 0.0078  -> wasted-group fraction
gd.group_size_for_epsilon(0.1, 0.1) # 14      -> G to reach 90% of the limit at p=0.1
gd.per_prompt_gradient(3, 8)        # 0.4841  -> this group's gradient gain
```

## Reproduce

```bash
uv run python scripts/checks.py             # numerical checks, incl. the general vector identity
uv run python scripts/grpo_diagnostics.py   # the closed forms as an API + a worked cheat sheet
uv run python scripts/experiment_dynamics.py# controlled GRPO run -> fig_experiment.pdf (CPU, no LLM)
uv run python scripts/digitize_dapo.py      # (optional) re-digitize DAPO Fig. 3b; needs the DAPO PDF (see below)
uv run python scripts/make_figures.py       # theory exhibits + DAPO consistency check
uv run python scripts/analyze_rollouts.py   # Big-Math solve rates -> real-data exhibit + tables
make                                         # figures + compile paper/main.pdf
```

Requires [`uv`](https://docs.astral.sh/uv/) and a LaTeX toolchain (`latexmk`).
The real-data script pulls the ungated mirror
`open-r1/Big-Math-RL-Verified-Processed` from the Hugging Face Hub. The digitized
DAPO curve (`data/dapo/dapo_fig3b_accuracy1_fraction.csv`) ships with the repo, so
`make_figures.py` runs without it; re-running `digitize_dapo.py` requires the
released DAPO paper PDF (arXiv:2503.14476) placed at
`data/dapo/dapo_2503.14476.pdf`, which is not redistributed here.

## Layout

```
paper/main.tex                  the note
paper/references.bib            bibliography
paper/figures/                  generated figure PDFs
paper/tables/                   generated LaTeX tables
scripts/checks.py               numerical checks (incl. general vector identity)
scripts/grpo_diagnostics.py     the closed forms as a tiny diagnostic API
scripts/experiment_dynamics.py  controlled GRPO/Dr.GRPO/DAPO run -> fig_experiment
scripts/figstyle.py             shared figure style
scripts/make_figures.py         theory exhibits + DAPO consistency check
scripts/analyze_rollouts.py     real-data exhibit + tables
scripts/digitize_dapo.py        digitize DAPO Fig. 3b discard curve
data/dapo/                      digitized DAPO discard curve (CSV; source in its header)
```

## License

The source code (`scripts/`, `Makefile`) is released under the MIT License
([LICENSE](LICENSE)). The paper text and figures (`paper/`) are licensed under
CC BY 4.0 ([paper/LICENSE](paper/LICENSE)).

## Citation

```bibtex
@misc{bay2026grpostd,
  title  = {GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number: The Group-Standard-Deviation Identity},
  author = {Bay, Yong Yi and Yearick, Kathleen A.},
  year   = {2026}
}
```
