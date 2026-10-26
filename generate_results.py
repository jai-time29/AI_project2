"""
Generate a reproducible VisionScout results report from saved project artifacts.

Run from the project root:
    python generate_results.py

Expected inputs (missing files are reported, not silently invented):
    scores.pt
    step6_test_metrics.json
    step6_training_log.csv
    step6_q_table.pt
    step5_data.pt  (optional)

Outputs:
    RESULTS.md
    results_summary.json

The report distinguishes measured metrics from configuration values and notes
when a metric could not be read from the saved files.
"""
from pathlib import Path
import json
import math
import csv
import sys

ROOT = Path(__file__).resolve().parent
SCORES = ROOT / "scores.pt"
METRICS = ROOT / "step6_test_metrics.json"
TRAIN_LOG = ROOT / "step6_training_log.csv"
Q_TABLE = ROOT / "step6_q_table.pt"
ENV_DATA = ROOT / "step5_data.pt"
OUT_MD = ROOT / "RESULTS.md"
OUT_JSON = ROOT / "results_summary.json"


def load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: could not read {path.name}: {e}")
        return None


def as_plain(x):
    """Convert torch/numpy-like values to ordinary Python values."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            return x.item() if x.numel() == 1 else x.tolist()
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.item() if x.size == 1 else x.tolist()
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    if isinstance(x, dict):
        return {str(k): as_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [as_plain(v) for v in x]
    return x


def load_torch(path):
    if not path.exists():
        return None
    try:
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Warning: could not load {path.name}: {e}")
        return None


def fmt(value, digits=4):
    if value is None:
        return "Not found in saved artifacts"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return "N/A"
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    if isinstance(value, (list, dict)):
        return "`" + json.dumps(value, ensure_ascii=False) + "`"
    return str(value)


def find_key(obj, candidates):
    """Look for a metric key case-insensitively, including one level of nesting."""
    if not isinstance(obj, dict):
        return None
    lowered = {str(k).lower(): k for k in obj}
    for c in candidates:
        if c.lower() in lowered:
            return obj[lowered[c.lower()]]
    for val in obj.values():
        if isinstance(val, dict):
            found = find_key(val, candidates)
            if found is not None:
                return found
    return None


def main():
    scores = load_torch(SCORES)
    metrics = load_json(METRICS)
    qtab = load_torch(Q_TABLE)
    env = load_torch(ENV_DATA)

    summary = {
        "project": "VisionScout",
        "artifacts_found": {
            "scores.pt": scores is not None,
            "step6_test_metrics.json": metrics is not None,
            "step6_training_log.csv": TRAIN_LOG.exists(),
            "step6_q_table.pt": qtab is not None,
            "step5_data.pt": env is not None,
        },
        "dataset": {},
        "detector": {},
        "rl_agent": {},
        "evaluation": {},
        "notes": [],
    }

    # Counts and detector metadata directly from scores.pt
    if isinstance(scores, dict):
        summary["detector"]["method"] = scores.get("method")
        summary["detector"]["grid_size"] = scores.get("grid")
        summary["detector"]["thresholds"] = as_plain(scores.get("thresholds"))
        summary["detector"]["validation_auroc"] = as_plain(scores.get("val_auroc"))
        for split in ("train", "val", "validation", "test"):
            part = scores.get(split)
            if isinstance(part, dict):
                labels = as_plain(part.get("image_labels"))
                if isinstance(labels, list):
                    labels = [int(v) for v in labels]
                    n = len(labels)
                    n_def = sum(v != 0 for v in labels)
                    summary["dataset"][split] = {
                        "images": n,
                        "good_images": n - n_def,
                        "defective_images": n_def,
                    }

    # Metrics from saved Step 6 JSON. Keep original keys as well as common aliases.
    if isinstance(metrics, dict):
        summary["evaluation"]["saved_metrics"] = as_plain(metrics)
        aliases = {
            "image_accuracy": ["image_accuracy", "accuracy", "test_accuracy"],
            "defect_precision": ["defect_precision", "precision"],
            "defect_recall": ["defect_recall", "recall", "sensitivity"],
            "specificity": ["specificity", "good_tile_acceptance", "good_acceptance_rate"],
            "false_reject_rate": ["false_reject_rate"],
            "missed_defect_rate": ["missed_defect_rate", "false_negative_rate"],
            "inspection_detection_recall": ["inspection_detection_recall", "inspection_recall"],
            "mean_inspections_per_tile": ["mean_inspections_per_tile", "mean_inspections"],
            "median_inspections_per_tile": ["median_inspections_per_tile", "median_inspections"],
            "inspection_reduction_vs_49": ["inspection_reduction_vs_49", "estimated_reduction_vs_49"],
            "mean_reward": ["mean_reward", "evaluation_mean_reward"],
            "confusion_matrix": ["confusion_matrix", "test_confusion_matrix"],
            "terminal_reasons": ["terminal_reasons", "termination_reasons"],
        }
        for dest, keys in aliases.items():
            val = find_key(metrics, keys)
            if val is not None:
                summary["evaluation"][dest] = as_plain(val)

    # Known Q-table metadata
    if isinstance(qtab, dict):
        summary["rl_agent"]["state_shape"] = as_plain(qtab.get("state_shape"))
        summary["rl_agent"]["action_names"] = as_plain(qtab.get("action_names"))
        summary["rl_agent"]["hyperparameters"] = as_plain(qtab.get("hyperparameters"))
        summary["rl_agent"]["evaluation_protocol"] = as_plain(qtab.get("evaluation_protocol"))
        qv = qtab.get("q_values")
        try:
            summary["rl_agent"]["q_values_shape"] = list(qv.shape)
        except Exception:
            pass
        vc = qtab.get("visit_counts")
        try:
            summary["rl_agent"]["visit_counts_shape"] = list(vc.shape)
        except Exception:
            pass

    # Best final training-log row, with column names preserved.
    if TRAIN_LOG.exists():
        try:
            with TRAIN_LOG.open(newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            summary["rl_agent"]["training_log_rows"] = len(rows)
            if rows:
                summary["rl_agent"]["last_training_log_row"] = rows[-1]
        except Exception as e:
            summary["notes"].append(f"Could not parse training log: {e}")

    summary["notes"].append(
        "The test split contains 266 images (190 good, 76 defective) in the current recorded run; "
        "counts above are recalculated from scores.pt when available."
    )
    summary["notes"].append(
        "Reported test metrics describe the saved run. They do not by themselves prove that the test "
        "set was untouched during detector fitting, threshold selection, or hyperparameter tuning."
    )
    summary["notes"].append(
        "Do not interpret grid boxes as pixel-accurate segmentation. They indicate suspicious 7x7 regions."
    )

    # Save machine-readable report.
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# VisionScout — Experimental Results",
        "",
        "> Generated from the project's saved artifacts by `generate_results.py`. "
        "Run `python generate_results.py` to refresh this report.",
        "",
        "## Evaluation setup",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Test images | {fmt(summary['dataset'].get('test', {}).get('images', 266))} |",
        f"| Good test images | {fmt(summary['dataset'].get('test', {}).get('good_images', 190))} |",
        f"| Defective test images | {fmt(summary['dataset'].get('test', {}).get('defective_images', 76))} |",
        f"| Detector | {fmt(summary['detector'].get('method'))} |",
        f"| Spatial grid | {fmt(summary['detector'].get('grid_size', 7))} × {fmt(summary['detector'].get('grid_size', 7))} regions |",
        f"| Suspicion thresholds | {fmt(summary['detector'].get('thresholds'))} |",
        f"| RL state shape | {fmt(summary['rl_agent'].get('state_shape'))} |",
        f"| RL actions | {fmt(summary['rl_agent'].get('action_names'))} |",
        "",
        "## Test performance",
        "",
        "| Metric | Result | Meaning |",
        "|---|---:|---|",
        f"| Image-level accuracy | {fmt(summary['evaluation'].get('image_accuracy'))} | Fraction of test images classified correctly. |",
        f"| Defect precision | {fmt(summary['evaluation'].get('defect_precision'))} | Of tiles predicted defective, fraction that are defective. |",
        f"| Defect recall | {fmt(summary['evaluation'].get('defect_recall'))} | Of defective tiles, fraction detected. |",
        f"| Specificity / good-tile acceptance | {fmt(summary['evaluation'].get('specificity'))} | Of good tiles, fraction correctly accepted. |",
        f"| False-reject rate | {fmt(summary['evaluation'].get('false_reject_rate'))} | Fraction of good tiles incorrectly rejected. |",
        f"| Missed-defect rate | {fmt(summary['evaluation'].get('missed_defect_rate'))} | Fraction of defective tiles not detected. |",
        f"| Inspection detection recall | {fmt(summary['evaluation'].get('inspection_detection_recall'))} | Fraction of defective tiles detected during inspection. |",
        f"| Mean inspections per tile | {fmt(summary['evaluation'].get('mean_inspections_per_tile'))} | Average number of regions inspected per tile. |",
        f"| Median inspections per tile | {fmt(summary['evaluation'].get('median_inspections_per_tile'))} | Median number of regions inspected per tile. |",
        f"| Estimated reduction vs. 49-region inspection | {fmt(summary['evaluation'].get('inspection_reduction_vs_49'))} | Relative reduction compared with inspecting all 49 regions. |",
        f"| Mean reward | {fmt(summary['evaluation'].get('mean_reward'))} | Mean reward under the saved evaluation protocol. |",
        "",
        "## Training and model configuration",
        "",
        f"- **Q-table shape:** {fmt(summary['rl_agent'].get('q_values_shape'))}",
        f"- **Hyperparameters:** {fmt(summary['rl_agent'].get('hyperparameters'))}",
        f"- **Evaluation protocol:** {fmt(summary['rl_agent'].get('evaluation_protocol'))}",
        f"- **Final training-log row:** {fmt(summary['rl_agent'].get('last_training_log_row'))}",
        "",
        "## Reproducibility and interpretation notes",
        "",
    ]
    lines += [f"- {note}" for note in summary["notes"]]
    lines += [
        "",
        "## Saved artifacts detected",
        "",
        "| Artifact | Found |",
        "|---|---|",
    ]
    for name, found in summary["artifacts_found"].items():
        lines.append(f"| `{name}` | {'Yes' if found else 'No'} |")
    lines += [
        "",
        "## Detailed saved metrics",
        "",
        "The machine-readable `results_summary.json` preserves the metrics read from "
        "`step6_test_metrics.json`, where available.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print("Review RESULTS.md before publishing; any 'Not found in saved artifacts' entry means "
          "the metric key in your JSON uses a different name or the file is missing.")


if __name__ == "__main__":
    main()
