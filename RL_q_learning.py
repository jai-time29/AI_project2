"""
VisionScout — Step 6: Robust tabular Q-learning policy
======================================================

Reads step5_data.pt produced by step5_environment.py. It does not refit the
anomaly detector and does not alter Step 4 scores or suspicion thresholds.

Training:
  * 20,000 Q-learning episodes by default
  * 50/50 good/defective image sampling via balanced_image_index
  * valid-action masking (the agent cannot choose an empty suspicion level)
  * terminal-aware Bellman targets
  * gradually decaying epsilon-greedy exploration
  * all policy decisions see only the environment's observation/state/reward

Evaluation:
  * each test image is evaluated exactly once
  * natural test class distribution is retained (no balancing)
  * reports classification, inspection-effort and terminal-reason metrics

Run in the project folder:
    python step6_q_learning.py

Expected files alongside this script:
    step5_environment.py
    step5_data.pt

Outputs:
    step6_q_table.pt
    step6_test_metrics.json
    step6_training_log.csv
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from step5_environment import (
    ACTION_NAMES,
    INSPECT_HIGH,
    INSPECT_MEDIUM,
    INSPECT_LOW,
    STOP_ACCEPT,
    STOP_REJECT,
    MAX_INSPECTIONS,
    InspectionEnv,
    balanced_image_index,
)

# ----------------------------- Configuration -----------------------------
DATA_FILE = Path("step5_data.pt")
MODEL_FILE = Path("step6_q_table.pt")
METRICS_FILE = Path("step6_test_metrics.json")
LOG_FILE = Path("step6_training_log.csv")

SEED = 42
EPISODES = 20_000
LEARNING_RATE = 0.10       # alpha
DISCOUNT = 0.95            # gamma
EPSILON_START = 1.00
EPSILON_MIN = 0.05
# Exponential schedule reaches approximately 0.056 at the last episode.
# This preserves exploration far longer than multiplying epsilon by 0.995
# every episode, which would hit the floor after only about 600 episodes.
EPSILON_DECAY_STEPS = EPISODES / 4.0
LOG_EVERY = 1_000
MOVING_WINDOW = 200

# State is (high_bin, medium_bin, low_bin, budget_bin): 3 x 3 x 3 x 5.
STATE_SHAPE = (3, 3, 3, 5)
N_ACTIONS = 5
# Stable tie breaking for evaluation: inspect the most suspicious available
# level first; if no inspection action is tied for best, prefer accept over
# an unsupported rejection. Q values still determine the winning value.
EVAL_TIE_PRIORITY = {
    INSPECT_HIGH: 0,
    INSPECT_MEDIUM: 1,
    INSPECT_LOW: 2,
    STOP_ACCEPT: 3,
    STOP_REJECT: 4,
}


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location="cpu")


def validate_data(data: dict) -> None:
    if not isinstance(data, dict) or "train" not in data or "test" not in data:
        raise TypeError(f"{DATA_FILE} must contain 'train' and 'test' dictionaries.")
    for name in ("train", "test"):
        split = data[name]
        required = ("image_labels", "region_labels", "scores", "levels")
        missing = [key for key in required if key not in split]
        if missing:
            raise KeyError(f"step5_data.pt['{name}'] missing keys: {missing}")
        image_labels = torch.as_tensor(split["image_labels"]).long().flatten()
        region_labels = torch.as_tensor(split["region_labels"]).long()
        scores = torch.as_tensor(split["scores"]).float()
        levels = torch.as_tensor(split["levels"]).long()
        expected = (len(image_labels), 49)
        if tuple(region_labels.shape) != expected:
            raise ValueError(f"{name}.region_labels must be {expected}, got {tuple(region_labels.shape)}")
        if tuple(scores.shape) != expected or tuple(levels.shape) != expected:
            raise ValueError(
                f"{name}.scores and levels must be {expected}; got "
                f"{tuple(scores.shape)} and {tuple(levels.shape)}"
            )
        if not torch.isin(image_labels, torch.tensor([0, 1])).all():
            raise ValueError(f"{name}.image_labels must be binary (0=good, 1=defective).")
        if not torch.isin(region_labels, torch.tensor([0, 1])).all():
            raise ValueError(f"{name}.region_labels must be binary.")
        if not torch.isin(levels, torch.tensor([0, 1, 2])).all():
            raise ValueError(f"{name}.levels must be 0=low, 1=medium, 2=high.")
        if not torch.isfinite(scores).all():
            raise ValueError(f"{name}.scores contain NaN or infinity.")
        if not torch.equal(region_labels.bool().any(dim=1), image_labels.bool()):
            raise ValueError(f"{name}: image and region labels disagree.")


def validate_state(state: tuple) -> Tuple[int, int, int, int]:
    if not isinstance(state, tuple) or len(state) != 4:
        raise ValueError(f"Unexpected environment state: {state!r}")
    s = tuple(int(x) for x in state)
    if not (0 <= s[0] < 3 and 0 <= s[1] < 3 and 0 <= s[2] < 3 and 0 <= s[3] < 5):
        raise ValueError(f"Environment state falls outside the expected bins: {s!r}")
    return s


def epsilon_at(episode: int) -> float:
    """Slow exponential decay from 1.0 toward the 0.05 exploration floor."""
    return EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * math.exp(
        -float(episode) / EPSILON_DECAY_STEPS
    )


def choose_action(q: np.ndarray, state: tuple, valid_actions: List[int],
                  epsilon: float, rng: random.Random, training: bool) -> int:
    """Epsilon-greedy selection over legal actions only."""
    if not valid_actions:
        raise RuntimeError("Cannot choose an action in a terminal state.")
    if training and rng.random() < epsilon:
        return rng.choice(valid_actions)

    vals = [float(q[state + (a,)]) for a in valid_actions]
    best_val = max(vals)
    tied = [a for a in valid_actions if np.isclose(float(q[state + (a,)]), best_val,
                                                    rtol=1e-6, atol=1e-8)]
    if training:
        return rng.choice(tied)
    return min(tied, key=lambda action: EVAL_TIE_PRIORITY[action])


def train_q_learning(train_split: dict, episodes: int, seed: int):
    py_rng = random.Random(seed)
    # Independent PRNG seed for reproducible torch-side behavior if needed.
    torch.manual_seed(seed)
    q = np.zeros(STATE_SHAPE + (N_ACTIONS,), dtype=np.float32)
    visits = np.zeros(STATE_SHAPE + (N_ACTIONS,), dtype=np.int64)
    env = InspectionEnv(train_split, train_split["levels"], max_inspections=MAX_INSPECTIONS)

    recent_rewards = deque(maxlen=MOVING_WINDOW)
    recent_inspections = deque(maxlen=MOVING_WINDOW)
    recent_correct = deque(maxlen=MOVING_WINDOW)
    log_rows = []

    print("\nTraining tabular Q-learning")
    print(f"  episodes={episodes:,}, alpha={LEARNING_RATE}, gamma={DISCOUNT}")
    print(f"  epsilon: {EPSILON_START:.2f} -> approximately {epsilon_at(episodes):.3f} "
          f"(minimum {EPSILON_MIN:.2f})")
    print("  training images sampled with approximately 50/50 good/defective balance")

    for ep in range(1, episodes + 1):
        idx = balanced_image_index(train_split)
        state = validate_state(env.reset(idx))
        total_reward = 0.0
        terminal_info = None
        epsilon = epsilon_at(ep - 1)

        for _ in range(MAX_INSPECTIONS + 1):
            valid = env.valid_actions()
            action = choose_action(q, state, valid, epsilon, py_rng, training=True)
            result = env.step(action)
            next_state = validate_state(result.state)
            reward = float(result.reward)
            total_reward += reward

            # Bellman target: do not bootstrap after a terminal transition;
            # otherwise bootstrap only over legal actions in the next state.
            if result.done:
                target = reward
                terminal_info = result.info
            else:
                next_valid = env.valid_actions()
                if not next_valid:
                    raise RuntimeError("Environment returned nonterminal state with no legal actions.")
                best_next = max(float(q[next_state + (a,)]) for a in next_valid)
                target = reward + DISCOUNT * best_next

            old_value = float(q[state + (action,)])
            visits[state + (action,)] += 1
            q[state + (action,)] = old_value + LEARNING_RATE * (target - old_value)
            state = next_state

            if result.done:
                break
        else:
            raise RuntimeError(f"Episode {ep} failed to terminate within the inspection budget.")

        if terminal_info is None:
            raise RuntimeError(f"Episode {ep} terminated without final transition metadata.")

        inspections = len(env.inspected)
        correct = int(bool(terminal_info.get("correct", False)))
        recent_rewards.append(total_reward)
        recent_inspections.append(inspections)
        recent_correct.append(correct)

        if ep % LOG_EVERY == 0 or ep == 1 or ep == episodes:
            row = {
                "episode": ep,
                "epsilon": float(epsilon),
                "mean_reward_last_window": float(np.mean(recent_rewards)),
                "mean_inspections_last_window": float(np.mean(recent_inspections)),
                "accuracy_last_window_balanced_train": float(np.mean(recent_correct)),
                "episodes_in_window": len(recent_rewards),
            }
            log_rows.append(row)
            print(
                f"  episode {ep:6,d}/{episodes:,} | eps={epsilon:.3f} | "
                f"reward={row['mean_reward_last_window']:7.2f} | "
                f"inspections={row['mean_inspections_last_window']:.2f} | "
                f"balanced-train acc={row['accuracy_last_window_balanced_train']:.3f}"
            )

    # This is a tabular policy; no gradient optimizer or neural network is needed.
    return q, log_rows, visits


def evaluate_policy(q: np.ndarray, test_split: dict) -> dict:
    """Greedy evaluation, one pass over each test image at natural prevalence."""
    env = InspectionEnv(test_split, test_split["levels"], max_inspections=MAX_INSPECTIONS)
    n = len(torch.as_tensor(test_split["image_labels"]).flatten())
    counts = Counter()
    total_reward = 0.0
    total_inspections = 0
    all_inspection_counts = []

    for idx in range(n):
        state = validate_state(env.reset(idx))
        episode_reward = 0.0
        terminal_info = None
        for _ in range(MAX_INSPECTIONS + 1):
            valid = env.valid_actions()
            action = choose_action(q, state, valid, epsilon=0.0,
                                  rng=random.Random(0), training=False)
            result = env.step(action)
            episode_reward += float(result.reward)
            state = validate_state(result.state)
            if result.done:
                terminal_info = result.info
                break
        if terminal_info is None:
            raise RuntimeError(f"Test image {idx} did not terminate.")

        y = int(terminal_info["true_image_label"])
        reason = str(terminal_info.get("terminal_reason", "unknown"))
        # Classification: found defect or explicit reject => defective prediction.
        predicted_bad = int(reason in ("defect_found", "stopped_reject"))
        counts["n"] += 1
        counts["positive"] += int(y == 1)
        counts["negative"] += int(y == 0)
        counts["tp"] += int(y == 1 and predicted_bad == 1)
        counts["fn"] += int(y == 1 and predicted_bad == 0)
        counts["tn"] += int(y == 0 and predicted_bad == 0)
        counts["fp"] += int(y == 0 and predicted_bad == 1)
        counts["found_defect"] += int(reason == "defect_found")
        counts["explicit_reject"] += int(reason == "stopped_reject")
        counts["accept"] += int(predicted_bad == 0)
        counts["terminal_" + reason] += 1
        total_reward += episode_reward
        total_inspections += len(env.inspected)
        all_inspection_counts.append(len(env.inspected))

    def ratio(num: float, den: float):
        return float(num / den) if den else None

    tp, fn, tn, fp = counts["tp"], counts["fn"], counts["tn"], counts["fp"]
    avg_inspections = ratio(total_inspections, n)
    metrics = {
        "test_images": n,
        "defective_images": counts["positive"],
        "good_images": counts["negative"],
        "defect_prevalence": ratio(counts["positive"], n),
        "image_accuracy": ratio(tp + tn, n),
        "defect_precision": ratio(tp, tp + fp),
        "defect_recall": ratio(tp, tp + fn),
        "specificity_good_tile_acceptance": ratio(tn, tn + fp),
        "false_reject_rate_on_good_tiles": ratio(fp, tn + fp),
        "missed_defect_rate_on_defective_tiles": ratio(fn, tp + fn),
        "inspection_detection_recall": ratio(counts["found_defect"], counts["positive"]),
        "true_positive": tp,
        "false_negative": fn,
        "true_negative": tn,
        "false_positive": fp,
        "tiles_with_defect_found_by_inspection": counts["found_defect"],
        "tiles_explicitly_rejected_without_finding_defect": counts["explicit_reject"],
        "tiles_accepted": counts["accept"],
        "mean_inspections_per_tile": avg_inspections,
        "median_inspections_per_tile": float(np.median(all_inspection_counts)) if n else None,
        "max_inspections_used": int(max(all_inspection_counts, default=0)),
        "fraction_of_49_regions_inspected": ratio(total_inspections, n * 49),
        "estimated_inspection_reduction_vs_49_regions": (
            1.0 - total_inspections / (n * 49) if n else None
        ),
        "mean_total_reward_per_tile": ratio(total_reward, n),
        "terminal_reasons": {
            key.removeprefix("terminal_"): int(value)
            for key, value in counts.items() if key.startswith("terminal_")
        },
        "evaluation_protocol": "greedy policy; every test image exactly once; natural test prevalence",
    }
    return metrics


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE.resolve()}. Run step5_environment.py first "
            "and keep step5_data.pt in this folder."
        )
    data = safe_torch_load(DATA_FILE)
    validate_data(data)

    train_labels = torch.as_tensor(data["train"]["image_labels"]).long().flatten()
    test_labels = torch.as_tensor(data["test"]["image_labels"]).long().flatten()
    print("VisionScout — Step 6 Q-learning")
    print(f"Loaded Step 5 data: {len(train_labels)} train images "
          f"({int((train_labels == 0).sum())} good, {int((train_labels == 1).sum())} defective); "
          f"{len(test_labels)} test images "
          f"({int((test_labels == 0).sum())} good, {int((test_labels == 1).sum())} defective).")
    print(f"Anomaly method carried from Step 5: {data.get('method', 'not recorded')}")
    print("No detector fitting, score recalculation, or threshold changes occur in Step 6.")

    q, log_rows, visits = train_q_learning(data["train"], EPISODES, SEED)

    # Evaluate only after training, and never use test results to tune the policy.
    print("\nEvaluating frozen greedy policy on the full natural-distribution test split...")
    metrics = evaluate_policy(q, data["test"])
    metrics["training"] = {
        "algorithm": "tabular Q-learning",
        "episodes": EPISODES,
        "alpha": LEARNING_RATE,
        "gamma": DISCOUNT,
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay_schedule": "epsilon_min + (epsilon_start - epsilon_min) * exp(-episode / (episodes/4))",
        "seed": SEED,
        "balanced_image_sampling": "approximately 50% good / 50% defective per training episode",
        "max_inspections": MAX_INSPECTIONS,
    }
    metrics["q_table"] = {
        "state_shape": list(STATE_SHAPE),
        "action_count": N_ACTIONS,
        "visited_state_action_pairs": int(np.count_nonzero(visits)),
        "total_q_updates": int(visits.sum()),
        "nonzero_states": int(np.count_nonzero(np.any(q != 0, axis=-1))),
    }

    torch.save({
        "q_values": torch.from_numpy(q.copy()),
        "visit_counts": torch.from_numpy(visits.copy()),
        "state_shape": STATE_SHAPE,
        "action_names": ACTION_NAMES,
        "hyperparameters": metrics["training"],
        "evaluation_protocol": metrics["evaluation_protocol"],
        "step5_method": data.get("method"),
        "step5_thresholds": data.get("thresholds"),
    }, MODEL_FILE)
    with LOG_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    with METRICS_FILE.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nTEST RESULTS (test split not used for learning or model selection)")
    for key in (
        "image_accuracy", "defect_precision", "defect_recall",
        "specificity_good_tile_acceptance", "false_reject_rate_on_good_tiles",
        "missed_defect_rate_on_defective_tiles", "inspection_detection_recall",
        "mean_inspections_per_tile", "median_inspections_per_tile",
        "estimated_inspection_reduction_vs_49_regions", "mean_total_reward_per_tile",
    ):
        value = metrics[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"  terminal_reasons: {metrics['terminal_reasons']}")
    print("\nSaved artifacts:")
    print(f"  Q-table / policy : {MODEL_FILE.resolve()}")
    print(f"  test metrics     : {METRICS_FILE.resolve()}")
    print(f"  training log     : {LOG_FILE.resolve()}")
    print("Step 6 complete. Review recall, false-reject rate, and inspection effort together; "
          "do not judge the policy on accuracy alone.")


if __name__ == "__main__":
    main()
