"""VisionScout plots. Run: python visualizations.py
Saves PNGs in ./figures without retraining anything."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scale plot typography to 75% of Matplotlib's usual defaults for README-friendly figures.
plt.rcParams.update({
    "font.size": 7.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.75,
    "ytick.labelsize": 6.75,
    "legend.fontsize": 7.5,
    "figure.titlesize": 9.0,
})

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# Recorded Step 4 validation AUROC results.
AUROC = {
    "Mean distance": (0.728, 0.696),
    "Gaussian / Mahalanobis": (0.795, 0.883),
    "kNN": (0.788, 0.855),
}
# Recorded Step 4 test-region summary.
LEVELS = {"Low": (10953, 1.8), "Medium": (1483, 9.8), "High": (598, 34.3)}
# Reconstructed from Step 6 test metrics: rows actual [good, defective],
# columns predicted [good, defective].
CM = np.array([[190, 0], [10, 66]])


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close()
    print("[saved]", OUT / name)


def load_pt(path):
    try:
        import torch
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[skip] {path.name}: {e}")
        return None


def npval(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        return np.asarray(x)
    except Exception:
        return None


def find_key(obj, keys):
    """Find first matching key in nested dictionaries."""
    if not isinstance(obj, dict):
        return None
    for k, v in obj.items():
        if str(k).lower() in keys:
            return v
    for v in obj.values():
        found = find_key(v, keys)
        if found is not None:
            return found
    return None


def auroc_plot():
    names = list(AUROC)
    x = np.arange(len(names))
    w = 0.36
    plt.figure(figsize=(9, 5))
    plt.bar(x-w/2, [AUROC[n][0] for n in names], w, label="Region AUROC")
    plt.bar(x+w/2, [AUROC[n][1] for n in names], w, label="Image AUROC")
    plt.axhline(.5, linestyle="--", linewidth=1, label="Random baseline")
    plt.xticks(x, names, rotation=10, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("AUROC")
    plt.title("Step 4: Validation AUROC Comparison")
    plt.legend()
    save("01_auroc_comparison.png")


def confusion_plot():
    plt.figure(figsize=(6, 5))
    plt.imshow(CM, interpolation="nearest")
    plt.colorbar(label="Number of tiles")
    labels = ["Good", "Defective"]
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Step 6: Q-learning Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(CM[i, j]), ha="center", va="center", fontsize=11.25)
    save("02_confusion_matrix.png")


def suspicion_plot():
    names = list(LEVELS)
    counts = [LEVELS[n][0] for n in names]
    rates = [LEVELS[n][1] for n in names]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, rates)
    plt.ylabel("Defective regions (%)")
    plt.title("Step 4: Defect Rate by Suspicion Level")
    plt.ylim(0, max(rates)*1.25)
    for b, r, n in zip(bars, rates, counts):
        plt.text(b.get_x()+b.get_width()/2, r, f"{r:.1f}% (n={n:,})",
                 ha="center", va="bottom")
    save("03_defect_rate_by_suspicion.png")


def efficiency_plot():
    labels = ["Inspect all 49", "Q-learning mean"]
    values = [49, 4.3571]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values)
    plt.ylabel("Regions inspected per tile")
    plt.title("Step 6: Inspection Effort")
    plt.ylim(0, 55)
    for b, v in zip(bars, values):
        plt.text(b.get_x()+b.get_width()/2, v+1, f"{v:.2f}", ha="center")
    plt.figtext(.5, .01, "Estimated inspection reduction: 91.11%", ha="center")
    save("04_inspection_efficiency.png")


def terminal_plot(metrics):
    reasons = {
        "Defect found": 65,
        "Stopped / accepted": 185,
        "Budget exhausted": 15,
        "Stopped / rejected": 1,
    }
    if isinstance(metrics, dict):
        raw = metrics.get("terminal_reasons", {})
        if isinstance(raw, dict):
            for key, val in raw.items():
                k = str(key).lower()
                if "defect_found" in k:
                    reasons["Defect found"] = int(val)
                elif "budget" in k:
                    reasons["Budget exhausted"] = int(val)
                elif "accept" in k:
                    reasons["Stopped / accepted"] = int(val)
                elif "reject" in k:
                    reasons["Stopped / rejected"] = int(val)
    plt.figure(figsize=(9, 5))
    bars = plt.bar(list(reasons), list(reasons.values()))
    plt.ylabel("Test tiles")
    plt.title("Step 6: Episode Terminal Outcomes")
    plt.xticks(rotation=10, ha="right")
    for b, v in zip(bars, reasons.values()):
        plt.text(b.get_x()+b.get_width()/2, v+1, str(v), ha="center")
    save("05_terminal_outcomes.png")


def training_plots():
    p = ROOT / "step6_training_log.csv"
    if not p.exists():
        print("[skip] step6_training_log.csv not found")
        return
    df = pd.read_csv(p)
    if df.empty:
        print("[skip] training log is empty")
        return
    print("[info] training log columns:", list(df.columns))
    cols = {str(c).lower(): c for c in df.columns}
    ep = next((cols[k] for k in cols if "episode" in k or k in ("ep", "step")), df.columns[0])
    rew = next((cols[k] for k in cols if "reward" in k), None)
    eps = next((cols[k] for k in cols if "epsilon" in k or k == "eps"), None)
    x = pd.to_numeric(df[ep], errors="coerce")
    if rew:
        y = pd.to_numeric(df[rew], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.any():
            plt.figure(figsize=(9, 5))
            plt.plot(x[ok], y[ok], alpha=.5, label="Recorded reward")
            plt.plot(x[ok], y[ok].rolling(max(3, min(10, len(y[ok])//10 or 3)),
                                         min_periods=1).mean(), label="Smoothed")
            plt.xlabel(ep); plt.ylabel(rew); plt.title("Step 6: Q-learning Training Reward")
            plt.legend(); save("06_training_reward.png")
        else:
            print("[skip] reward values not numeric")
    else:
        print("[skip] no reward column in training log")
    if eps:
        y = pd.to_numeric(df[eps], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.any():
            plt.figure(figsize=(9, 5))
            plt.plot(x[ok], y[ok])
            plt.xlabel(ep); plt.ylabel("Epsilon"); plt.title("Step 6: Epsilon Decay")
            save("07_epsilon_decay.png")
        else:
            print("[skip] epsilon values not numeric")
    else:
        print("[skip] no epsilon column in training log")


def q_table_plot():
    p = ROOT / "step6_q_table.pt"
    if not p.exists():
        print("[skip] step6_q_table.pt not found")
        return
    obj = load_pt(p)
    if obj is None:
        return
    q = find_key(obj, {"q_table", "qtable", "q", "values"}) if isinstance(obj, dict) else obj
    if q is None:
        print("[skip] Q table key not recognized; top-level keys:", list(obj) if isinstance(obj, dict) else type(obj))
        return
    arr = np.squeeze(npval(q))
    if arr is None:
        print("[skip] cannot convert Q-table")
        return
    if arr.ndim == 5 and arr.shape[-1] == 5:
        arr = arr.reshape(-1, 5)
    if arr.ndim != 2:
        print("[skip] unexpected Q-table shape:", arr.shape)
        return
    if arr.shape[1] != 5 and arr.shape[0] == 5:
        arr = arr.T
    if arr.shape[1] != 5:
        print("[skip] expected 5 actions; Q table shape:", arr.shape)
        return
    plt.figure(figsize=(10, 7))
    plt.imshow(arr, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Q-value")
    plt.xlabel("Action (0=high, 1=medium, 2=low, 3=accept, 4=reject)")
    plt.ylabel("Encoded state")
    plt.title("Step 6: Learned Q-table")
    save("08_q_table_heatmap.png")


def score_and_roc_plots():
    p = ROOT / "scores.pt"
    if not p.exists():
        print("[skip] scores.pt not found")
        return
    obj = load_pt(p)
    if obj is None:
        return
    # A true ROC/PR curve needs raw per-example scores plus binary labels.
    scores = find_key(obj, {"image_scores", "image_score", "image_anomaly_scores"})
    labels = find_key(obj, {"image_labels", "image_label", "gt_image_labels"})
    if scores is not None and labels is not None:
        s, y = np.ravel(npval(scores)), np.ravel(npval(labels))
        if len(s) == len(y) and len(np.unique(y)) == 2:
            try:
                from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
                fpr, tpr, _ = roc_curve(y, s)
                plt.figure(figsize=(7, 6))
                plt.plot(fpr, tpr, label=f"AUROC={auc(fpr,tpr):.3f}")
                plt.plot([0,1], [0,1], "--", label="Random")
                plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
                plt.title("Image-level ROC Curve"); plt.legend()
                save("09_image_roc_curve.png")
                pr, rc, _ = precision_recall_curve(y, s)
                plt.figure(figsize=(7, 6))
                plt.plot(rc, pr, label=f"Average precision={average_precision_score(y,s):.3f}")
                plt.xlabel("Recall"); plt.ylabel("Precision")
                plt.title("Image-level Precision–Recall Curve"); plt.legend()
                save("10_image_precision_recall_curve.png")
            except Exception as e:
                print("[skip] ROC/PR:", e)
        else:
            print("[skip] image scores and labels have incompatible shapes")
    else:
        print("[skip] no recognizable image score/label pair in scores.pt; not inventing ROC/PR curves")

    # Optional region score histogram if the saved file has region scores and labels.
    rs = find_key(obj, {"region_scores", "region_score", "anomaly_scores"})
    ry = find_key(obj, {"region_labels", "gt_region_labels"})
    if rs is not None and ry is not None:
        s, y = np.ravel(npval(rs)), np.ravel(npval(ry))
        if len(s) == len(y):
            ok = np.isfinite(s) & np.isfinite(y)
            s, y = s[ok], y[ok]
            if len(s) and len(np.unique(y)) == 2:
                plt.figure(figsize=(9, 5))
                plt.hist(s[y == 0], bins=50, alpha=.6, density=True, label="Normal")
                plt.hist(s[y == 1], bins=50, alpha=.6, density=True, label="Defective")
                plt.axvline(44.712, linestyle="--", label="Threshold 1")
                plt.axvline(72.323, linestyle="--", label="Threshold 2")
                plt.xlabel("Anomaly score"); plt.ylabel("Density")
                plt.title("Step 4: Anomaly Score Distribution"); plt.legend()
                save("11_anomaly_score_distribution.png")
            else:
                print("[skip] region score labels do not contain both classes")
        else:
            print("[skip] region score and label lengths differ")
    else:
        print("[skip] raw region scores/labels not recognized in scores.pt")


def main():
    print("VisionScout visualization generator")
    metrics = None
    mp = ROOT / "step6_test_metrics.json"
    if mp.exists():
        try:
            metrics = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as e:
            print("[warn] could not read metrics JSON:", e)
    auroc_plot()
    confusion_plot()
    suspicion_plot()
    efficiency_plot()
    terminal_plot(metrics)
    training_plots()
    q_table_plot()
    score_and_roc_plots()
    print(f"\nFinished. Generated plots are in: {OUT}")
    print("Skipped plots need raw arrays/fields that were not found in the saved artifacts.")


if __name__ == "__main__":
    main()
