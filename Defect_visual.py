"""
VisionScout - image-level defect visualisations.

Uses ONLY saved data: scores.pt (Step 4) + the original images and masks.
Region k of the 7x7 grid is row k // 7, column k % 7 (same order as
fmap.flatten(2) in precompute_features.py). Images were resized to 224x224
without cropping, so each grid cell maps to an (H/7 x W/7) box of the original.

Outputs (in figures/):
  defect_<type>_<file>.png   4-panel figure per selected test image
  comparison_grid.png        good vs defective examples side by side

Run from the project root:  python Defect_visual.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

SCORES_FILE = "scores.pt"
OUT_DIR = "figures"
LEVEL_COLORS = {1: "orange", 2: "red"}      # medium / high (low regions are left unboxed)
LEVEL_NAMES = {0: "low", 1: "medium", 2: "high"}


def to_np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def load_test():
    s = torch.load(SCORES_FILE, map_location="cpu", weights_only=False)
    t = s["test"]
    return {
        "scores": to_np(t["scores"]).astype(float),          # [N, 49]
        "levels": to_np(t["levels"]).astype(int),            # [N, 49]
        "region_labels": to_np(t["region_labels"]),          # [N, 49]
        "image_labels": to_np(t["image_labels"]).astype(int),
        "defect_type": list(t["defect_type"]),
        "paths": [os.path.normpath(p) for p in t["paths"]],
        "thresholds": to_np(s["thresholds"]).astype(float),
        "grid": int(s["grid"]),
        "method": s.get("method", "?"),
    }


def load_image_and_mask(path):
    img = np.array(Image.open(path).convert("L"))
    mask_path = path[:-4] + ".png"
    mask = np.array(Image.open(mask_path).convert("L")) > 0 if os.path.exists(mask_path) \
        else np.zeros_like(img, dtype=bool)
    return img, mask


def show_img(ax, img):
    """Draw image in pixel coordinates (0..W, 0..H) so boxes/contours line up exactly."""
    H, W = img.shape
    ax.imshow(img, cmap="gray", extent=(0, W, H, 0))


def draw_gt_contour(ax, mask):
    if mask.any():
        H, W = mask.shape
        ax.contour(np.arange(W) + 0.5, np.arange(H) + 0.5, mask.astype(float),
                   levels=[0.5], colors="cyan", linewidths=1)


def fix_limits(ax, H, W):
    ax.set_xlim(0, W); ax.set_ylim(H, 0)


def draw_level_boxes(ax, levels, G, H, W):
    ch, cw = H / G, W / G
    for k, lv in enumerate(levels):
        if lv in LEVEL_COLORS:
            r, c = divmod(k, G)
            ax.add_patch(patches.Rectangle((c * cw, r * ch), cw, ch, fill=False,
                                           edgecolor=LEVEL_COLORS[lv], linewidth=2))


def plot_single(d, i, vmax):
    G = d["grid"]
    img, mask = load_image_and_mask(d["paths"][i])
    H, W = img.shape
    sc = d["scores"][i].reshape(G, G)
    gt = d["region_labels"][i].reshape(G, G) > 0
    lv = d["levels"][i]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    title = (f"{d['defect_type'][i]}  |  {os.path.basename(d['paths'][i])}  |  "
             f"max score {sc.max():.1f}  |  high regions: {(lv == 2).sum()}, medium: {(lv == 1).sum()}")
    fig.suptitle(title, fontsize=11)

    show_img(axes[0], img)
    axes[0].set_title("Original")

    hm = axes[1].imshow(sc, cmap="inferno", vmin=0, vmax=vmax)
    rr, cc = np.nonzero(gt)
    axes[1].scatter(cc, rr, marker="x", c="cyan", s=40, label="GT defect region")
    axes[1].set_title(f"7x7 anomaly scores ({d['method']})")
    if gt.any():
        axes[1].legend(loc="lower right", fontsize=7)
    cb = fig.colorbar(hm, ax=axes[1], fraction=0.046)
    for t in d["thresholds"]:
        cb.ax.axhline(t, color="white", lw=1, ls="--")

    show_img(axes[2], img)
    axes[2].imshow(sc, cmap="inferno", vmin=0, vmax=vmax, alpha=0.45,
                   extent=(0, W, H, 0), interpolation="nearest")
    axes[2].set_title("Score overlay")

    show_img(axes[3], img)
    draw_level_boxes(axes[3], lv, G, H, W)
    draw_gt_contour(axes[3], mask)
    for ax in (axes[0], axes[2], axes[3]):
        fix_limits(ax, H, W)
    axes[3].set_title("Suspicious regions (orange=med, red=high)\ncyan = GT defect outline", fontsize=9)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    name = f"defect_{d['defect_type'][i]}_{os.path.splitext(os.path.basename(d['paths'][i]))[0]}.png"
    fig.savefig(os.path.join(OUT_DIR, name), dpi=130)
    plt.close(fig)
    return name


def select_examples(d):
    """Per defect type: clearest detection (highest max score) and hardest case
    (lowest max score). For normal tiles: cleanest and most false-alarm-like."""
    max_s = d["scores"].max(1)
    picks = []
    for t in sorted(set(d["defect_type"])):
        idx = [i for i, dt in enumerate(d["defect_type"]) if dt == t]
        idx = sorted(idx, key=lambda i: max_s[i])
        if t == "MT_Free":
            picks += [("good: cleanest", idx[0]), ("good: most suspicious", idx[-1])]
        else:
            picks += [(f"{t}: clearest", idx[-1])]
            if len(idx) > 1:
                picks += [(f"{t}: hardest", idx[0])]
    return picks


def plot_comparison(d, vmax):
    G = d["grid"]
    rows = []
    for t, cap in [("MT_Free", "Good tile"), ("MT_Blowhole", "Defect: blowhole"),
                   ("MT_Crack", "Defect: crack"), ("MT_Break", "Defect: break")]:
        idx = [i for i, dt in enumerate(d["defect_type"]) if dt == t]
        if idx:
            # median-score example = a typical case, not a cherry-picked one
            idx = sorted(idx, key=lambda i: d["scores"][i].max())
            rows.append((cap, idx[len(idx) // 2]))

    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.2 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (cap, i) in enumerate(rows):
        img, mask = load_image_and_mask(d["paths"][i])
        H, W = img.shape
        sc = d["scores"][i].reshape(G, G)
        show_img(axes[r, 0], img)
        axes[r, 0].set_ylabel(cap, fontsize=11)
        show_img(axes[r, 1], img)
        axes[r, 1].imshow(sc, cmap="inferno", vmin=0, vmax=vmax, alpha=0.45,
                          extent=(0, W, H, 0), interpolation="nearest")
        show_img(axes[r, 2], img)
        draw_level_boxes(axes[r, 2], d["levels"][i], G, H, W)
        draw_gt_contour(axes[r, 2], mask)
        for ax in axes[r]:
            fix_limits(ax, H, W)
            ax.set_xticks([]); ax.set_yticks([])
    for c, t in enumerate(["Original", "Score overlay", "Suspicious regions + GT (cyan)"]):
        axes[0, c].set_title(t)
    fig.suptitle("VisionScout: good vs defective test tiles (typical cases)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "comparison_grid.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    d = load_test()
    vmax = float(np.percentile(d["scores"], 99.5))   # same colour scale for every figure
    print(f"Loaded {len(d['paths'])} test images, method={d['method']}, "
          f"thresholds={np.round(d['thresholds'], 2).tolist()}")

    for cap, i in select_examples(d):
        name = plot_single(d, i, vmax)
        print(f"  {cap:28s} -> figures/{name}")

    plot_comparison(d, vmax)
    print("  comparison grid            -> figures/comparison_grid.png")
