"""Numerical checks of the closed forms used in the paper.

For binary rewards and a Bernoulli-logit prompt (p = sigmoid(theta), score
d/dtheta log pi = y - p) we compare Monte-Carlo estimates against the closed
forms:

    standardized advantage      A_+ = sqrt((G-k)/k),  A_- = -sqrt(k/(G-k))
    GRPO per-prompt gradient     E[g]   -> sqrt(p(1-p))         (arcsine VST)
    REINFORCE per-prompt grad    E[g]   =  p(1-p)(1 - 1/G)      (raw success)
    exact finite-group gradient  g(k)   =  sqrt(k(G-k))/G
    unanimous-group rate         P      =  p^G + (1-p)^G

Run: ``python scripts/checks.py``.
"""

import numpy as np


def grpo_advantage(r, normalize=True):
    mu = r.mean()
    if not normalize:
        return r - mu
    sd = r.std()
    return (r - mu) / sd if sd > 0 else np.zeros_like(r)


def mc_gradient(p, G, n, rng, normalize=True):
    g = np.empty(n)
    for j in range(n):
        y = (rng.random(G) < p).astype(float)
        g[j] = np.mean(grpo_advantage(y, normalize) * (y - p))
    return g.mean()


def main():
    rng = np.random.default_rng(0)

    # closed-form advantages
    for k, G in [(1, 4), (2, 8), (3, 16), (7, 10)]:
        r = np.array([1.0] * k + [0.0] * (G - k))
        A = grpo_advantage(r)
        assert np.isclose(A[0], np.sqrt((G - k) / k))
        assert np.isclose(A[-1], -np.sqrt(k / (G - k)))

    # exact finite-group identity g(k) = sqrt(k(G-k))/G
    for G in (5, 8, 16):
        for k in range(1, G):
            r = np.array([1.0] * k + [0.0] * (G - k))
            g = np.mean(grpo_advantage(r) * (r - 0.3))   # baseline arbitrary
            assert np.isclose(g, np.sqrt(k * (G - k)) / G)

    # GENERAL vector identity (any policy, any dimension):
    #   g = (1/G) sum_i A_i s_i = sigma * (sbar_+ - sbar_-),
    # where s_i are arbitrary per-sample score vectors, sbar_+ / sbar_- the mean
    # score of the correct / incorrect samples, and sigma = sqrt(k(G-k))/G.
    maxerr = 0.0
    for _ in range(5000):
        G = int(rng.integers(2, 20))
        k = int(rng.integers(1, G))
        d = int(rng.integers(1, 8))
        r = np.array([1.0] * k + [0.0] * (G - k))
        A = grpo_advantage(r)
        S = rng.normal(size=(G, d))                  # arbitrary score vectors
        g = (A[:, None] * S).mean(0)
        sigma = np.sqrt(k * (G - k)) / G
        rhs = sigma * (S[:k].mean(0) - S[k:].mean(0))
        maxerr = max(maxerr, float(np.max(np.abs(g - rhs))))
    assert maxerr < 1e-12, maxerr
    print(f"general vector identity max abs err = {maxerr:.1e}")

    # exact expectation vs delta-method, and where the delta method BREAKS:
    # near the extremes the approximation 1-1/(8Gp(1-p)) departs from the exact
    # binomial expectation (the unanimous mass is no longer negligible).
    def exact_expected_gradient(p, G):
        ks = np.arange(1, G)
        from math import comb
        c = np.array([comb(G, int(k)) for k in ks], float)
        return (c * p**ks * (1 - p)**(G - ks) * np.sqrt(ks * (G - ks))).sum() / G

    G = 8
    print(f"{'p':>5} {'exact frac':>11} {'delta-method':>13} {'abs gap':>9}")
    for p in (0.5, 0.25, 0.1, 0.05):
        exact = exact_expected_gradient(p, G) / np.sqrt(p * (1 - p))
        delta = 1 - 1 / (8 * G * p * (1 - p))
        print(f"{p:>5.2f} {exact:>11.4f} {delta:>13.4f} {abs(exact-delta):>9.4f}")
    # interior: delta method is tight; extreme: it is not
    assert abs(exact_expected_gradient(0.5, G) / 0.5 - (1 - 1/(8*G*0.25))) < 0.01
    assert (exact_expected_gradient(0.05, G) / np.sqrt(0.05*0.95)
            < (1 - 1/(8*G*0.05*0.95)) - 0.1)   # delta method overstates near 0

    # the group-size law: gradient fidelity and required G (pins the paper's
    # Table values so they cannot silently drift).
    from math import lgamma

    def fidelity(G, p):
        ks = np.arange(1, G)
        logc = lgamma(G + 1) - np.array([lgamma(k + 1) + lgamma(G - k + 1)
                                         for k in ks])
        terms = np.exp(logc + ks * np.log(p) + (G - ks) * np.log1p(-p))
        return (terms * np.sqrt(ks * (G - ks))).sum() / G / np.sqrt(p * (1 - p))

    def required_G(p, target, Gmax=20000):
        for G in range(2, Gmax):
            if fidelity(G, p) >= target:
                return G
        return None

    # fidelity matches the headline readings used in the text
    assert np.isclose(fidelity(8, 0.5), 0.9278, atol=5e-4)
    assert np.isclose(fidelity(8, 0.05), 0.5390, atol=5e-4)
    # exact required G vs closed-form G* = 1/(8 eps p(1-p)); table entries
    table = {(0.50, 0.10): 7, (0.50, 0.05): 11, (0.50, 0.01): 51,
             (0.30, 0.10): 9, (0.30, 0.05): 14, (0.30, 0.01): 61,
             (0.10, 0.10): 22, (0.10, 0.05): 36, (0.10, 0.01): 144,
             (0.05, 0.10): 42, (0.05, 0.05): 69, (0.05, 0.01): 273}
    for (p, eps), G_exact in table.items():
        assert required_G(p, 1 - eps) == G_exact, (p, eps, required_G(p, 1 - eps))
    # closed form is exact in the high-fidelity regime, optimistic at small-G
    # extremes: G* understates the exact requirement near p in {0,1}.
    assert 1 / (8 * 0.01 * 0.05 * 0.95) < table[(0.05, 0.01)]       # 263 < 273
    assert 1 / (8 * 0.10 * 0.05 * 0.95) < table[(0.05, 0.10)] - 10  # 26 << 42
    print("group-size law: fidelity and required-G table verified")

    # expected gradients vs closed forms
    G, n = 64, 200_000
    print(f"{'p':>5} {'GRPO mc':>9} {'sqrt(p(1-p))':>13} "
          f"{'RF mc':>9} {'p(1-p)(1-1/G)':>14}")
    for p in (0.1, 0.25, 0.5, 0.75, 0.9):
        gg = mc_gradient(p, G, n, rng, True)
        gr = mc_gradient(p, G, n, rng, False)
        tg, tr = np.sqrt(p * (1 - p)), p * (1 - p) * (1 - 1 / G)
        print(f"{p:>5.2f} {gg:>9.4f} {tg:>13.4f} {gr:>9.4f} {tr:>14.4f}")
        assert abs(gr - tr) < 2e-3
        assert abs(gg - tg) < 0.02

    # unanimous-group rate
    for p, G in [(0.5, 8), (0.1, 8), (0.05, 32)]:
        y = rng.random((100_000, G)) < p
        emp = np.mean((y.sum(1) == 0) | (y.sum(1) == G))
        assert abs(emp - (p**G + (1 - p)**G)) < 0.01

    print("all checks passed")


if __name__ == "__main__":
    main()
