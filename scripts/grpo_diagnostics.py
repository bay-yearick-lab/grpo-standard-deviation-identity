"""GRPO finite-group diagnostics: the paper's closed forms as a tiny API.

Every quantity below is read off a single group's correct-count ``k`` (out of
``G``) or off a per-prompt success probability ``p``; none of them needs a model,
a training run, or a baseline. The functions are the practitioner-facing form of
the group-standard-deviation identity and its corollaries.

Cheat sheet
-----------
==============================  ===================================  ============
quantity                        closed form                          function
==============================  ===================================  ============
per-prompt gradient (scalar     sqrt(k(G-k)) / G  = sigma            per_prompt_gradient
  gain of the update)
silent-group rate               p**G + (1-p)**G                      silent_rate
expected per-prompt gradient     (1/G) sum_k C(G,k) p^k (1-p)^(G-k)  expected_gradient
  (exact, finite G)                * sqrt(k(G-k))
finite-G attenuation             1 - 1/(8 G p(1-p))  + O(G^-2)        attenuation
group-size budget                G >= 1 / (8 eps p(1-p))              group_size_for_epsilon
difficulty weight (arcsine)      1 / sqrt(p(1-p))                     difficulty_weight
gradient-mass weight             GRPO: sqrt(p(1-p));  raw: p(1-p)     mass_weight
==============================  ===================================  ============

Run ``python scripts/grpo_diagnostics.py`` to print a worked cheat-sheet table.
"""

from __future__ import annotations

from math import comb

import numpy as np

__all__ = [
    "per_prompt_gradient",
    "silent_rate",
    "expected_gradient",
    "attenuation",
    "group_size_for_epsilon",
    "difficulty_weight",
    "mass_weight",
    "mass_share",
]


def per_prompt_gradient(k, G):
    """Exact per-prompt GRPO gradient gain for a group with ``k`` of ``G`` correct.

    Returns ``sqrt(k(G-k)) / G``, which equals the group's empirical reward
    standard deviation ``sigma`` and is the scalar coefficient of the update
    (the group-standard-deviation identity). Zero for a unanimous group.
    """
    k = np.asarray(k, dtype=float)
    return np.sqrt(np.clip(k * (G - k), 0.0, None)) / G


def silent_rate(p, G):
    """Probability a size-``G`` group is silent (unanimous, zero advantage)."""
    p = np.asarray(p, dtype=float)
    return p ** G + (1.0 - p) ** G


def expected_gradient(p, G):
    """Exact expectation of the per-prompt gradient over k ~ Binomial(G, p).

    The finite sum ``(1/G) sum_{k=1}^{G-1} C(G,k) p^k (1-p)^(G-k) sqrt(k(G-k))``.
    This is exact (no asymptotics); ``attenuation`` is its large-G expansion.
    """
    p = np.asarray(p, dtype=float)
    ks = np.arange(1, G)
    coef = np.array([comb(G, int(k)) for k in ks], dtype=float)
    root = np.sqrt(ks * (G - ks))
    # outer over a (possibly array) p
    pk = p[..., None] ** ks
    qk = (1.0 - p[..., None]) ** (G - ks)
    return (coef * root * pk * qk).sum(-1) / G


def attenuation(p, G):
    """Large-G attenuation factor 1 - 1/(8 G p(1-p)) (asymptotic, Corollary)."""
    p = np.asarray(p, dtype=float)
    return 1.0 - 1.0 / (8.0 * G * p * (1.0 - p))


def group_size_for_epsilon(eps, p):
    """Smallest G keeping the per-prompt gradient within fraction ``eps`` of the limit.

    Inverts the attenuation: ``G >= 1 / (8 eps p(1-p))``.
    """
    p = np.asarray(p, dtype=float)
    return np.ceil(1.0 / (8.0 * eps * p * (1.0 - p))).astype(int)


def difficulty_weight(p):
    """Arcsine difficulty weight w(p) = d/dp [2 arcsin sqrt p] = 1/sqrt(p(1-p))."""
    p = np.asarray(p, dtype=float)
    return 1.0 / np.sqrt(p * (1.0 - p))


def mass_weight(p, normalize_std=True):
    """Per-prompt gradient-mass weight: GRPO sqrt(p(1-p)); raw/Dr.GRPO p(1-p)."""
    p = np.asarray(p, dtype=float)
    base = p * (1.0 - p)
    return np.sqrt(base) if normalize_std else base


def mass_share(p, mask, normalize_std=True):
    """Share of total gradient budget that prompts in ``mask`` receive."""
    w = mass_weight(np.asarray(p, dtype=float), normalize_std)
    mask = np.asarray(mask, dtype=bool)
    return float(w[mask].sum() / w.sum())


def _cheatsheet():
    G = 8
    print(f"Group-standard-deviation identity, worked at G={G}\n")
    print(f"{'k':>3} {'g(k)=sqrt(k(G-k))/G':>22}")
    for k in range(0, G + 1):
        print(f"{k:>3} {float(per_prompt_gradient(k, G)):>22.4f}")
    print()
    print(f"{'p':>6} {'silent(G=8)':>12} {'E[g] exact':>12} "
          f"{'atten':>8} {'w(p)':>7} {'G@eps=.1':>9}")
    for p in (0.05, 0.25, 0.5, 0.75, 0.95):
        print(f"{p:>6.2f} {float(silent_rate(p, 8)):>12.4f} "
              f"{float(expected_gradient(np.array(p), 8)):>12.4f} "
              f"{float(attenuation(np.array(p), 8)):>8.4f} "
              f"{float(difficulty_weight(p)):>7.3f} "
              f"{int(group_size_for_epsilon(0.1, np.array(p))):>9d}")


if __name__ == "__main__":
    _cheatsheet()
