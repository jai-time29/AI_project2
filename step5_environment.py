"""
VisionScout — Step 5: MDP inspection environment
================================================
Expected inputs:
  features.pt  (frozen region features and ground-truth labels)
  scores.pt    (finalized Step 4 scores and suspicion levels)
with dictionaries data["train"] and data["test"], each containing:
  feats:         [N, 49, 512] tensor
  region_labels: [N, 49] tensor (1 = defective region)
  image_labels:  [N] tensor (1 = defective tile)
  paths / defect_type (optional metadata)

This script loads the finalized Step 4 anomaly scores and suspicion levels,
defines the inspection MDP, and runs smoke tests. It does NOT recalculate
scores or train Q-learning; Q-learning is Step 6.

Run:
  python step5_environment.py

Reward design chosen with the user:
  inspect region:             -0.5
  discover a defect:          +20 (episode ends; tile rejected)
  stop and accept good tile:  +5
  stop and accept bad tile:   -100
  stop and reject bad tile:   +20
  stop and reject good tile:  -30
  maximum inspections:        20
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch


# -------------------------- Configuration --------------------------

INPUT_FILE = Path("features.pt")
SCORES_FILE = Path("scores.pt")
OUTPUT_FILE = Path("step5_data.pt")

SEED = 42
GRID_REGIONS = 49
N_LEVELS = 3
CROSSFIT_FOLDS = 5
RIDGE = 1e-3

# Thresholds are quantiles of honest, out-of-fold normal-training scores.
# Regions <= q80 are low; q80..q95 are medium; > q95 are high.
LEVEL_QUANTILES = (0.80, 0.95)

INSPECTION_COST = -0.5
FOUND_DEFECT_REWARD = 20.0
ACCEPT_GOOD_REWARD = 5.0
MISS_DEFECT_PENALTY = -100.0
REJECT_DEFECT_REWARD = 20.0
FALSE_REJECT_PENALTY = -30.0
MAX_INSPECTIONS = 20

# State bins: high, medium, low remaining regions, inspections already used.
HIGH_BINS = (0, 1, 2)       # 0, 1, 2, 3+
MEDIUM_BINS = (0, 1, 2)     # 0, 1-2, 3+
LOW_BINS = (0, 1, 2)        # 0, 1-5, 6+
BUDGET_BINS = (0, 1, 2, 3, 4)  # 0, 1-5, 6-10, 11-15, 16-20

# Actions
INSPECT_HIGH = 0
INSPECT_MEDIUM = 1
INSPECT_LOW = 2
STOP_ACCEPT = 3
STOP_REJECT = 4
ACTION_NAMES = {
    INSPECT_HIGH: "inspect_high",
    INSPECT_MEDIUM: "inspect_medium",
    INSPECT_LOW: "inspect_low",
    STOP_ACCEPT: "stop_accept",
    STOP_REJECT: "stop_reject",
}


def load_pt(path: Path):
    """Load a local PyTorch file, including common older PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_split(name: str, split: dict) -> None:
    required = ("feats", "region_labels", "image_labels")
    missing = [k for k in required if k not in split]
    if missing:
        raise KeyError(
            f"{name} in {INPUT_FILE} is missing {missing}. "
            "Expected feats, region_labels, and image_labels."
        )

    feats = torch.as_tensor(split["feats"]).float()
    rlabels = torch.as_tensor(split["region_labels"])
    ilabels = torch.as_tensor(split["image_labels"]).flatten()

    if feats.ndim != 3 or feats.shape[1] != GRID_REGIONS:
        raise ValueError(
            f"{name}.feats should have shape [N, 49, D], got {tuple(feats.shape)}"
        )
    if rlabels.shape != feats.shape[:2]:
        raise ValueError(
            f"{name}.region_labels should have shape {tuple(feats.shape[:2])}, "
            f"got {tuple(rlabels.shape)}"
        )
    if len(ilabels) != len(feats):
        raise ValueError(f"{name}.image_labels count does not match feats count.")
    if not torch.isfinite(feats).all():
        raise ValueError(f"{name}.feats contains NaN or infinity.")
    if not torch.isin(rlabels, torch.tensor([0, 1])).all():
        raise ValueError(f"{name}.region_labels must contain only 0/1.")
    if not torch.isin(ilabels, torch.tensor([0, 1])).all():
        raise ValueError(f"{name}.image_labels must contain only 0/1.")

    # Defective image labels should agree with the presence of at least one
    # positive region label. This catches mask/label inconsistencies early.
    region_has_defect = rlabels.bool().any(dim=1)
    mismatch = region_has_defect != ilabels.bool()
    if mismatch.any():
        idx = mismatch.nonzero().flatten()[:10].tolist()
        raise ValueError(
            f"{name}: image_labels and region_labels disagree for image indices "
            f"{idx}. Check masks and the four empty-mask MT_Uneven images."
        )


def fit_gaussian(normal_regions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a regularized Gaussian model and return mean and precision matrix.
    normal_regions: [number_of_regions, feature_dim]
    """
    x = normal_regions.float()
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("Need at least two normal region feature vectors.")
    mean = x.mean(dim=0)
    xc = x - mean
    cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)

    # Ridge regularization stabilizes covariance inversion.
    scale = torch.trace(cov) / cov.shape[0]
    cov = cov + torch.eye(cov.shape[0]) * max(float(scale) * RIDGE, 1e-6)
    precision = torch.linalg.pinv(cov, hermitian=True)
    return mean, precision


def mahalanobis_scores(feats: torch.Tensor, mean: torch.Tensor,
                       precision: torch.Tensor) -> torch.Tensor:
    """Return squared Mahalanobis score for every region: [N, 49]."""
    n, r, d = feats.shape
    x = feats.reshape(-1, d).float()
    delta = x - mean
    scores = torch.einsum("bi,ij,bj->b", delta, precision, delta)
    return scores.reshape(n, r).clamp_min_(0)


def prepare_scores(data: dict) -> Tuple[dict, dict, torch.Tensor]:
    """
    Compute train/test scores from normal training images only.
    Normal train images receive 5-fold out-of-fold scores for calibration,
    avoiding unrealistically low in-sample scores for those images.
    """
    tr, te = data["train"], data["test"]
    tr_feats = torch.as_tensor(tr["feats"]).float()
    te_feats = torch.as_tensor(te["feats"]).float()
    tr_labels = torch.as_tensor(tr["image_labels"]).long().flatten()
    normal_idx = (tr_labels == 0).nonzero().flatten()

    if len(normal_idx) < CROSSFIT_FOLDS:
        raise ValueError(
            f"Need at least {CROSSFIT_FOLDS} defect-free training images "
            "for cross-fitting."
        )

    generator = torch.Generator().manual_seed(SEED)
    shuffled = normal_idx[torch.randperm(len(normal_idx), generator=generator)]
    folds = [f for f in shuffled.chunk(min(CROSSFIT_FOLDS, len(shuffled))) if len(f)]

    # Final model fitted on all normal training images.
    normal_train_regions = tr_feats[normal_idx].reshape(-1, tr_feats.shape[-1])
    mean, precision = fit_gaussian(normal_train_regions)
    train_scores = mahalanobis_scores(tr_feats, mean, precision)
    test_scores = mahalanobis_scores(te_feats, mean, precision)

    # Replace scores for normal training images with out-of-fold scores.
    for fold_i, held in enumerate(folds, start=1):
        held_set = set(held.tolist())
        others = torch.tensor(
            [int(i) for i in normal_idx.tolist() if int(i) not in held_set],
            dtype=torch.long,
        )
        if len(others) == 0:
            raise ValueError("Cross-fit fold has no remaining normal training images.")
        fold_normal = tr_feats[others].reshape(-1, tr_feats.shape[-1])
        fmean, fprecision = fit_gaussian(fold_normal)
        train_scores[held] = mahalanobis_scores(tr_feats[held], fmean, fprecision)
        print(f"  cross-fit fold {fold_i}/{len(folds)} complete")

    calibration = train_scores[normal_idx].flatten()
    thresholds = torch.quantile(
        calibration, torch.tensor(LEVEL_QUANTILES, dtype=calibration.dtype)
    )
    print(
        "Normal-score means: train cross-fit = "
        f"{train_scores[normal_idx].mean():.3f}, "
        f"test normal = {test_scores[torch.as_tensor(te['image_labels']).long() == 0].mean():.3f}"
    )
    print(f"Suspicion thresholds (q80, q95): {thresholds.tolist()}")

    return (
        {"scores": train_scores, "thresholds": thresholds},
        {"scores": test_scores, "thresholds": thresholds},
        thresholds,
    )


def score_to_levels(scores: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """0 = low, 1 = medium, 2 = high."""
    return torch.bucketize(scores.contiguous(), thresholds.contiguous())


def bin_high(n: int) -> int:
    return 0 if n == 0 else 1 if n == 1 else 2


def bin_medium(n: int) -> int:
    return 0 if n == 0 else 1 if n <= 2 else 2


def bin_low(n: int) -> int:
    return 0 if n == 0 else 1 if n <= 5 else 2


def bin_budget(used: int) -> int:
    if used == 0:
        return 0
    if used <= 5:
        return 1
    if used <= 10:
        return 2
    if used <= 15:
        return 3
    return 4


@dataclass
class StepResult:
    state: tuple
    reward: float
    done: bool
    info: dict


class InspectionEnv:
    """
    One episode = one tile.
    The agent sees only aggregate suspicion counts and inspection-budget usage.
    Ground-truth labels are revealed only for the region actually inspected.
    """

    def __init__(self, split: dict, levels: torch.Tensor, max_inspections: int = MAX_INSPECTIONS):
        self.split = split
        self.levels = levels.long()
        self.max_inspections = max_inspections
        self.feats = torch.as_tensor(split["feats"])
        self.region_labels = torch.as_tensor(split["region_labels"]).long()
        self.image_labels = torch.as_tensor(split["image_labels"]).long().flatten()
        if self.levels.shape != self.region_labels.shape:
            raise ValueError(
                f"levels shape {tuple(self.levels.shape)} does not match region labels "
                f"{tuple(self.region_labels.shape)}"
            )
        self.n_images = len(self.image_labels)
        self.reset(0)

    def reset(self, image_index: int | None = None) -> tuple:
        self.image_index = (
            random.randrange(self.n_images) if image_index is None else int(image_index)
        )
        if not 0 <= self.image_index < self.n_images:
            raise IndexError("image_index is outside this split.")
        self.true_image_label = int(self.image_labels[self.image_index])
        self.scores = torch.as_tensor(self.split["scores"])[self.image_index]
        self.level = self.levels[self.image_index]
        self.labels = self.region_labels[self.image_index]
        self.remaining = {
            lev: set((self.level == lev).nonzero().flatten().tolist())
            for lev in (0, 1, 2)
        }
        self.inspected = []
        self.done = False
        self.found_defect = False
        return self.state()

    def state(self) -> tuple:
        # Four components: binned high, medium, low counts and used budget.
        return (
            bin_high(len(self.remaining[2])),
            bin_medium(len(self.remaining[1])),
            bin_low(len(self.remaining[0])),
            bin_budget(len(self.inspected)),
        )

    def valid_actions(self) -> List[int]:
        if self.done:
            return []
        actions = [STOP_ACCEPT, STOP_REJECT]
        if len(self.inspected) < self.max_inspections:
            if self.remaining[2]:
                actions.append(INSPECT_HIGH)
            if self.remaining[1]:
                actions.append(INSPECT_MEDIUM)
            if self.remaining[0]:
                actions.append(INSPECT_LOW)
        return actions

    def _terminal_accept_reward(self) -> float:
        return ACCEPT_GOOD_REWARD if self.true_image_label == 0 else MISS_DEFECT_PENALTY

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is over; call reset() before another step.")
        if action not in self.valid_actions():
            raise ValueError(
                f"Invalid action {action} ({ACTION_NAMES.get(action, 'unknown')}); "
                f"valid actions are {[ACTION_NAMES[a] for a in self.valid_actions()]}"
            )

        before = self.state()
        info = {"action": ACTION_NAMES[action], "inspected_region": None}

        if action == STOP_ACCEPT:
            reward = self._terminal_accept_reward()
            self.done = True
            info.update(
                terminal_reason="stopped_accept",
                true_image_label=self.true_image_label,
                correct=(self.true_image_label == 0),
            )
            return StepResult(before, reward, True, info)

        if action == STOP_REJECT:
            reward = (
                REJECT_DEFECT_REWARD
                if self.true_image_label == 1
                else FALSE_REJECT_PENALTY
            )
            self.done = True
            info.update(
                terminal_reason="stopped_reject",
                true_image_label=self.true_image_label,
                correct=(self.true_image_label == 1),
            )
            return StepResult(before, reward, True, info)

        # Choose the highest-scoring remaining region within the selected level.
        level_id = {INSPECT_LOW: 0, INSPECT_MEDIUM: 1, INSPECT_HIGH: 2}[action]
        region = max(self.remaining[level_id], key=lambda i: float(self.scores[i]))
        self.remaining[level_id].remove(region)
        self.inspected.append(region)
        info["inspected_region"] = int(region)
        reward = INSPECTION_COST

        if int(self.labels[region]) == 1:
            self.found_defect = True
            reward += FOUND_DEFECT_REWARD
            self.done = True
            info.update(
                terminal_reason="defect_found",
                true_image_label=self.true_image_label,
                correct=True,
            )
            return StepResult(before, reward, True, info)

        # No defect was found. If the budget is exhausted, the policy must
        # accept; terminal reward reflects whether that was correct.
        if len(self.inspected) >= self.max_inspections:
            reward += self._terminal_accept_reward()
            self.done = True
            info.update(
                terminal_reason="budget_exhausted_accept",
                true_image_label=self.true_image_label,
                correct=(self.true_image_label == 0),
            )
            return StepResult(self.state(), reward, True, info)

        return StepResult(self.state(), reward, False, info)


def balanced_image_index(split: dict) -> int:
    """Sample a training image with approximately 50/50 image-level labels."""
    labels = torch.as_tensor(split["image_labels"]).long().flatten()
    good = (labels == 0).nonzero().flatten().tolist()
    bad = (labels == 1).nonzero().flatten().tolist()
    if not good or not bad:
        return random.randrange(len(labels))
    return random.choice(good if random.random() < 0.5 else bad)


def report_level_rates(split: dict, levels: torch.Tensor, name: str) -> None:
    labels = torch.as_tensor(split["region_labels"]).bool()
    print(f"\n{name} region-level defect rates:")
    for level_id, label in ((0, "low"), (1, "medium"), (2, "high")):
        m = levels == level_id
        total = int(m.sum())
        defective = int((labels & m).sum())
        rate = 100.0 * defective / max(total, 1)
        print(f"  {label:6s}: {defective:6d}/{total:6d} defective regions ({rate:.2f}%)")


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    for path in (INPUT_FILE, SCORES_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {path.resolve()}. Put step5_environment.py, "
                "features.pt, and scores.pt in the same folder and run it there."
            )

    features = load_pt(INPUT_FILE)
    scored = load_pt(SCORES_FILE)
    for name, obj in (("features.pt", features), ("scores.pt", scored)):
        if not isinstance(obj, dict) or "train" not in obj or "test" not in obj:
            raise TypeError(f"{name} must contain 'train' and 'test' dictionaries.")

    if scored.get("method") != "gauss":
        print(f"WARNING: scores.pt method is {scored.get('method')!r}, not 'gauss'.")
    thresholds = torch.as_tensor(scored.get("thresholds", []), dtype=torch.float32).flatten()
    if thresholds.numel() != 2:
        raise ValueError(f"Expected two saved Step 4 thresholds; got {thresholds.tolist()}.")

    # Use Step 4 scores/levels as-is. Do not fit another detector or recalculate thresholds.
    data = features
    for name in ("train", "test"):
        fsplit, ssplit = features[name], scored[name]
        validate_split(name, fsplit)
        for key in ("scores", "levels", "region_labels", "image_labels"):
            if key not in ssplit:
                raise KeyError(f"scores.pt['{name}'] is missing required key {key!r}.")

        scores = torch.as_tensor(ssplit["scores"]).float()
        levels = torch.as_tensor(ssplit["levels"]).long()
        rlabels = torch.as_tensor(ssplit["region_labels"]).long()
        ilabels = torch.as_tensor(ssplit["image_labels"]).long().flatten()
        f_rlabels = torch.as_tensor(fsplit["region_labels"]).long()
        f_ilabels = torch.as_tensor(fsplit["image_labels"]).long().flatten()

        expected_shape = (len(fsplit["image_labels"]), GRID_REGIONS)
        if tuple(scores.shape) != expected_shape or tuple(levels.shape) != expected_shape:
            raise ValueError(
                f"scores.pt {name} scores/levels must have shape {expected_shape}; "
                f"got {tuple(scores.shape)} and {tuple(levels.shape)}."
            )
        if not torch.equal(rlabels, f_rlabels) or not torch.equal(ilabels, f_ilabels):
            raise ValueError(
                f"Label mismatch between features.pt and scores.pt for {name}. "
                "Check that both files came from the same Step 4 feature split."
            )
        if not torch.isfinite(scores).all():
            raise ValueError(f"scores.pt {name} scores contain NaN or infinity.")
        if not torch.isin(levels, torch.tensor([0, 1, 2])).all():
            raise ValueError(f"scores.pt {name} levels must be 0=low, 1=medium, 2=high.")

        # Check image order when both files contain paths.
        fp, sp = fsplit.get("paths"), ssplit.get("paths")
        if fp is not None and sp is not None:
            fp = [str(x) for x in fp]
            sp = [str(x) for x in sp]
            if fp != sp:
                raise ValueError(
                    f"Image path/order mismatch between features.pt and scores.pt for {name}. "
                    "Do not train on mismatched splits."
                )

        fsplit["scores"] = scores
        fsplit["levels"] = levels
        # Keep Step 4's labels as the source of truth after equality is verified.
        fsplit["region_labels"] = rlabels
        fsplit["image_labels"] = ilabels

    print(f"Loaded {len(data['train']['image_labels'])} training images and "
          f"{len(data['test']['image_labels'])} test images.")
    print(f"Using Step 4 method: {scored.get('method')}; saved thresholds: {thresholds.tolist()}")
    print("Scores and levels loaded directly from scores.pt; no detector refit performed.")

    data["thresholds"] = thresholds
    data["method"] = scored.get("method")
    data["step5_config"] = {
        "inspection_cost": INSPECTION_COST,
        "found_defect_reward": FOUND_DEFECT_REWARD,
        "accept_good_reward": ACCEPT_GOOD_REWARD,
        "miss_defect_penalty": MISS_DEFECT_PENALTY,
        "reject_defect_reward": REJECT_DEFECT_REWARD,
        "false_reject_penalty": FALSE_REJECT_PENALTY,
        "max_inspections": MAX_INSPECTIONS,
        "seed": SEED,
        "scores_source": str(SCORES_FILE),
    }

    report_level_rates(data["train"], data["train"]["levels"], "TRAIN")
    report_level_rates(data["test"], data["test"]["levels"], "TEST")

    # Environment smoke test: verify actions, state shape, and termination.
    env = InspectionEnv(data["train"], data["train"]["levels"])
    state = env.reset(balanced_image_index(data["train"]))
    print("\nEnvironment smoke test:")
    print(f"  initial state = {state}")
    print(f"  valid actions = {[ACTION_NAMES[a] for a in env.valid_actions()]}")
    total_reward = 0.0
    n_steps = 0
    while not env.done and n_steps < MAX_INSPECTIONS + 2:
        valid = env.valid_actions()
        action = INSPECT_HIGH if INSPECT_HIGH in valid else STOP_ACCEPT
        result = env.step(action)
        total_reward += result.reward
        n_steps += 1
        print(
            f"  step {n_steps:02d}: action={result.info['action']}, "
            f"reward={result.reward:.1f}, next_state={result.state}, "
            f"done={result.done}"
        )
    if not env.done:
        raise RuntimeError("Smoke test failed: episode did not terminate.")
    print(f"  total reward = {total_reward:.1f}; episode terminated correctly.")

    torch.save(data, OUTPUT_FILE)
    print(f"\nSaved Step 5 environment data to {OUTPUT_FILE.resolve()}")
    print("Step 5 environment is ready. Step 6 will implement Q-learning separately.")


if __name__ == "__main__":
    main()
