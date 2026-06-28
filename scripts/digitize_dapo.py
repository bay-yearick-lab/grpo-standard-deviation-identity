"""Digitize DAPO's discarded-prompt fraction from the published figure.

Reads the released DAPO paper PDF (arXiv:2503.14476), renders Figure 3b -- "the
proportion of samples with an accuracy of 1" (avg@32 = 100%) versus training step
-- and extracts the curve by pixel detection of the plotted line, calibrated
against the printed axis tick marks. The accuracy-of-1 fraction is the all-correct
branch (k = G) of the silent-group mass p^G + (1-p)^G that DAPO's dynamic sampling
removes. DAPO trains with group size G = 16; the plotted metric uses n = 32 samples
per prompt. No model inference is performed: the curve comes only from the
published figure.

Output: data/dapo/dapo_fig3b_accuracy1_fraction.csv (step, all-correct fraction).
Run: uv run python scripts/digitize_dapo.py
"""

import csv
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAPO = os.path.join(HERE, "data", "dapo")
PDF = os.path.join(DAPO, "dapo_2503.14476.pdf")

# Axis tick anchors read from the rendered crop fig3b_full.png (600 dpi).
# Pixel centers of the printed ticks and their data values.
X_TICK_PIX = [390.5, 721.5, 1051.5, 1382.5, 1712.5]
X_TICK_VAL = [0, 2000, 4000, 6000, 8000]
Y_TICK_PIX = [425.0, 572.0, 718.5, 865.5, 1012.0, 1158.5, 1305.5]
Y_TICK_VAL = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
# Plot box interior (pixels), from the detected black spines.
X_LO, X_HI = 320, 1961
Y_LO, Y_HI = 322, 1352


def render_crop():
    """Render page 5 at 600 dpi and crop the Figure 3b panel (with axes)."""
    import subprocess

    base = os.path.join(DAPO, "page5_hi")
    subprocess.run(
        ["pdftoppm", "-png", "-r", "600", "-f", "5", "-l", "5", PDF, base],
        check=True,
    )
    im = Image.open(base + "-05.png").convert("RGB")
    W, H = im.size
    crop = im.crop((int(0.49 * W), int(0.085 * H),
                    int(0.93 * W), int(0.31 * H)))
    out = os.path.join(DAPO, "fig3b_full.png")
    crop.save(out)
    return out


def digitize(crop_path):
    a = np.asarray(Image.open(crop_path).convert("RGB")).astype(int)
    R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    # The plotted line is a light matplotlib blue; isolate it.
    blue = (B > 130) & (B - R > 30) & (B - G > 15) & (R < 200)

    mx, bx = np.polyfit(X_TICK_PIX, X_TICK_VAL, 1)
    my, by = np.polyfit(Y_TICK_PIX, Y_TICK_VAL, 1)

    cols, vals = [], []
    for px in range(X_LO, X_HI + 1):
        ys = np.where(blue[Y_LO:Y_HI + 1, px])[0]
        if len(ys) == 0:
            continue
        ymid = ys.mean() + Y_LO
        cols.append(mx * px + bx)
        vals.append(my * ymid + by)
    cols, vals = np.array(cols), np.array(vals)

    # Bin into evenly spaced step points; median per bin is robust to jitter.
    nb = 60
    edges = np.linspace(cols.min(), cols.max(), nb + 1)
    steps, frac = [], []
    for i in range(nb):
        m = (cols >= edges[i]) & (cols < edges[i + 1])
        if m.sum() == 0:
            continue
        steps.append(0.5 * (edges[i] + edges[i + 1]))
        frac.append(np.median(vals[m]))
    return np.array(steps), np.clip(np.array(frac), 0, None)


def main():
    crop = os.path.join(DAPO, "fig3b_full.png")
    if not os.path.exists(crop):
        crop = render_crop()
    steps, frac = digitize(crop)

    out = os.path.join(DAPO, "dapo_fig3b_accuracy1_fraction.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# Source: DAPO (Yu et al. 2025), arXiv:2503.14476, Figure 3b"])
        w.writerow(["# 'The proportion of samples with an accuracy of 1' "
                    "(avg@32 = 100%) vs training step."])
        w.writerow(["# All-correct (k=G) branch of the silent-group mass "
                    "p^G + (1-p)^G removed by dynamic sampling."])
        w.writerow(["# DAPO training group size G = 16; metric uses n = 32 "
                    "samples per prompt."])
        w.writerow(["# Digitized by pixel extraction; axis calibration RMS "
                    "< 2 steps (x), < 0.0002 (y)."])
        w.writerow(["step", "allcorrect_fraction"])
        for s, v in zip(steps, frac):
            w.writerow([int(round(s)), round(float(v), 4)])
    print(f"wrote {out}: {len(steps)} points, "
          f"step {int(steps.min())}-{int(steps.max())}, "
          f"frac {frac.min():.3f}-{frac.max():.3f}")


if __name__ == "__main__":
    main()
