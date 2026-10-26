# VisionScout — Experimental Results

> Generated from the project's saved artifacts by `generate_results.py`. Run `python generate_results.py` to refresh this report.

## Evaluation setup

| Item | Value |
|---|---|
| Test images | 266 |
| Good test images | 190 |
| Defective test images | 76 |
| Detector | gauss |
| Spatial grid | 7 × 7 regions |
| Suspicion thresholds | `[44.71189498901367, 72.32334899902344]` |
| RL state shape | `[3, 3, 3, 5]` |
| RL actions | `{"0": "inspect_high", "1": "inspect_medium", "2": "inspect_low", "3": "stop_accept", "4": "stop_reject"}` |

## Test performance

| Metric | Result | Meaning |
|---|---:|---|
| Image-level accuracy | 0.9624 | Fraction of test images classified correctly. |
| Defect precision | 1.0000 | Of tiles predicted defective, fraction that are defective. |
| Defect recall | 0.8684 | Of defective tiles, fraction detected. |
| Specificity / good-tile acceptance | Not found in saved artifacts | Of good tiles, fraction correctly accepted. |
| False-reject rate | Not found in saved artifacts | Fraction of good tiles incorrectly rejected. |
| Missed-defect rate | Not found in saved artifacts | Fraction of defective tiles not detected. |
| Inspection detection recall | 0.8553 | Fraction of defective tiles detected during inspection. |
| Mean inspections per tile | 4.3571 | Average number of regions inspected per tile. |
| Median inspections per tile | 1.0000 | Median number of regions inspected per tile. |
| Estimated reduction vs. 49-region inspection | Not found in saved artifacts | Relative reduction compared with inspecting all 49 regions. |
| Mean reward | Not found in saved artifacts | Mean reward under the saved evaluation protocol. |

## Training and model configuration

- **Q-table shape:** `[3, 3, 3, 5, 5]`
- **Hyperparameters:** `{"algorithm": "tabular Q-learning", "episodes": 20000, "alpha": 0.1, "gamma": 0.95, "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_decay_schedule": "epsilon_min + (epsilon_start - epsilon_min) * exp(-episode / (episodes/4))", "seed": 42, "balanced_image_sampling": "approximately 50% good / 50% defective per training episode", "max_inspections": 20}`
- **Evaluation protocol:** greedy policy; every test image exactly once; natural test prevalence
- **Final training-log row:** `{"episode": "20000", "epsilon": "0.06740333726370668", "mean_reward_last_window": "0.5325", "mean_inspections_last_window": "6.335", "accuracy_last_window_balanced_train": "0.86", "episodes_in_window": "200"}`

## Reproducibility and interpretation notes

- The test split contains 266 images (190 good, 76 defective) in the current recorded run; counts above are recalculated from scores.pt when available.
- Reported test metrics describe the saved run. They do not by themselves prove that the test set was untouched during detector fitting, threshold selection, or hyperparameter tuning.
- Do not interpret grid boxes as pixel-accurate segmentation. They indicate suspicious 7x7 regions.

## Saved artifacts detected

| Artifact | Found |
|---|---|
| `scores.pt` | Yes |
| `step6_test_metrics.json` | Yes |
| `step6_training_log.csv` | Yes |
| `step6_q_table.pt` | Yes |
| `step5_data.pt` | Yes |

## Detailed saved metrics

The machine-readable `results_summary.json` preserves the metrics read from `step6_test_metrics.json`, where available.
