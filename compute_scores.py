"""
VisionScout - Step 4: Suspiciousness score for every region.

Compares three anomaly-scoring methods, all fitted ONLY on defect-free tiles:
  1. mean   - Euclidean distance to the average normal region vector (simple baseline)
  2. gauss  - Mahalanobis distance to a Gaussian fitted per grid position (PaDiM-style)
  3. knn    - distance to the k nearest normal region vectors (PatchCore-style)

The best method is chosen by region-level AUROC on a validation set built from
the TRAIN split only (the test split is never used for choices), then scores are
bucketed into 3 levels (0 = low, 1 = medium, 2 = high) for the Q-learning state.

Input:  features.pt  (from Step 3)
Output: scores.pt, score_histogram.png
Run:    python compute_scores.py
"""
import torch

FEATURES = "features.pt"
OUT_FILE = "scores.pt"
SEED = 42
CALIB_FRAC = 0.2          # 20% of normal train tiles held out for validation + thresholds
KNN_K = 3
SHRINK = 0.01             # covariance regularisation for the Gaussian method
LEVEL_PCTS = (0.90, 0.99) # thresholds = 90th and 99th percentile of normal-region scores


# ----------------------------------------------------------------- helpers
def auroc(scores, labels):
    """Area under ROC curve via the rank formula (no sklearn needed)."""
    scores, labels = scores.flatten().double(), labels.flatten().bool()
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = torch.empty_like(scores)
    ranks[scores.argsort()] = torch.arange(1, len(scores) + 1, dtype=torch.double)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ----------------------------------------------------------------- methods
class MeanDistance:
    name = "mean"

    def fit(self, normal_feats):                       # [M, 49, 512]
        self.mu = normal_feats.reshape(-1, normal_feats.shape[-1]).mean(0)
        return self

    def score(self, feats):                            # [N, 49, 512] -> [N, 49]
        return (feats - self.mu).norm(dim=-1)


class PositionGaussian:
    name = "gauss"

    def fit(self, normal_feats):
        x = normal_feats.double()                      # [M, 49, D]
        M, P, D = x.shape
        self.mu = x.mean(0)                            # [49, D]
        xc = x - self.mu
        cov = torch.einsum("mpd,mpe->pde", xc, xc) / (M - 1)          # [49, D, D]
        cov += SHRINK * cov.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(D, dtype=torch.double)
        self.L = torch.linalg.cholesky(cov)            # [49, D, D]
        return self

    def score(self, feats):
        xc = (feats.double() - self.mu).permute(1, 2, 0)                # [49, D, N]
        z = torch.linalg.solve_triangular(self.L, xc, upper=False)      # whitened
        return z.pow(2).sum(1).sqrt().T.float()                         # [N, 49]


class KNNDistance:
    name = "knn"

    def fit(self, normal_feats):
        self.bank = normal_feats.reshape(-1, normal_feats.shape[-1])    # [M*49, D]
        return self

    def score(self, feats, chunk=1024):
        N, P, D = feats.shape
        q = feats.reshape(-1, D)
        out = []
        for i in range(0, len(q), chunk):
            d = torch.cdist(q[i:i + chunk], self.bank)                  # [chunk, bank]
            out.append(d.topk(KNN_K, largest=False).values.mean(1))
        return torch.cat(out).reshape(N, P)


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    data = torch.load(FEATURES, weights_only=False)
    tr, te = data["train"], data["test"]
    tr_feats, te_feats = tr["feats"].float(), te["feats"].float()

    # Split normal TRAIN tiles: "fit" (method learns normality) / "calib" (validation)
    g = torch.Generator().manual_seed(SEED)
    normal_idx = (tr["image_labels"] == 0).nonzero().flatten()
    normal_idx = normal_idx[torch.randperm(len(normal_idx), generator=g)]
    n_calib = int(len(normal_idx) * CALIB_FRAC)
    calib_idx, fit_idx = normal_idx[:n_calib], normal_idx[n_calib:]
    defect_idx = (tr["image_labels"] == 1).nonzero().flatten()
    val_idx = torch.cat([calib_idx, defect_idx])       # unseen normals + defective tiles
    print(f"Fit on {len(fit_idx)} normal tiles; validate on {len(calib_idx)} normal "
          f"+ {len(defect_idx)} defective train tiles")

    # Compare methods on validation
    results = {}
    print(f"\n{'method':8s} {'region AUROC':>13s} {'image AUROC':>12s}")
    for method in [MeanDistance(), PositionGaussian(), KNNDistance()]:
        method.fit(tr_feats[fit_idx])
        s = method.score(tr_feats[val_idx])
        r_auc = auroc(s, tr["region_labels"][val_idx])
        i_auc = auroc(s.max(1).values, tr["image_labels"][val_idx])
        results[method.name] = (method, r_auc, i_auc)
        print(f"{method.name:8s} {r_auc:13.3f} {i_auc:12.3f}")

    best_name = max(results, key=lambda k: results[k][1])
    best = results[best_name][0]
    print(f"\nBest method on validation: {best_name}")

    # ---- Score everything with the best method, using cross-fitting ----
    MethodClass = type(best)
    all_normal = (tr["image_labels"] == 0).nonzero().flatten()

    # Test tiles + defective train tiles: model fitted on ALL normal train tiles
    final = MethodClass().fit(tr_feats[all_normal])
    te_scores = final.score(te_feats)
    tr_scores = final.score(tr_feats)        # normal-tile rows get replaced below

    # Normal train tiles: 5-fold cross-fitting, so no tile is scored by a model that saw it
    K_FOLDS = 5
    perm = all_normal[torch.randperm(len(all_normal), generator=g)]
    folds = perm.chunk(K_FOLDS)
    for k, held in enumerate(folds):
        others = torch.cat([f for j, f in enumerate(folds) if j != k])
        tr_scores[held] = MethodClass().fit(tr_feats[others]).score(tr_feats[held])
        print(f"  cross-fit fold {k + 1}/{K_FOLDS} done")

    # Sanity check: honest train-normal scores should now look like test-normal scores
    te_normal = te_scores[te["image_labels"] == 0]
    print(f"Mean normal-tile score: train (cross-fit) = {tr_scores[all_normal].mean():.2f}, "
          f"test = {te_normal.mean():.2f}")

    # Level thresholds from honest (out-of-fold) normal-region scores
    calib_scores = tr_scores[all_normal].flatten()
    thresholds = torch.quantile(calib_scores, torch.tensor(LEVEL_PCTS))
    to_levels = lambda s: torch.bucketize(s.contiguous(), thresholds)   # 0 / 1 / 2

    print(f"Level thresholds: {[round(float(t), 3) for t in thresholds]}")
    print(f"TEST region AUROC = {auroc(te_scores, te['region_labels']):.3f}, "
          f"image AUROC = {auroc(te_scores.max(1).values, te['image_labels']):.3f}")

    te_lv, te_rl = to_levels(te_scores), te["region_labels"].bool()
    for lv, name in enumerate(["low", "medium", "high"]):
        m = te_lv == lv
        frac = te_rl[m].float().mean() if m.any() else torch.tensor(0.)
        print(f"  TEST level {lv} ({name:6s}): {int(m.sum()):6d} regions, "
              f"{100 * frac:5.1f}% are defective")

    def pack(split, scores):
        return {"scores": scores, "levels": to_levels(scores),
                "region_labels": split["region_labels"], "image_labels": split["image_labels"],
                "defect_type": split["defect_type"], "paths": split["paths"]}

    torch.save({"train": pack(tr, tr_scores), "test": pack(te, te_scores),
                "method": best_name, "thresholds": thresholds, "grid": data["grid"],
                "fit_idx": fit_idx, "calib_idx": calib_idx,
                "val_auroc": {k: v[1] for k, v in results.items()}}, OUT_FILE)
    print(f"\nSaved {OUT_FILE}")

    # Histogram: normal vs defective test regions
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.hist(te_scores[~te_rl].numpy(), bins=60, alpha=0.6, density=True, label="normal regions")
        plt.hist(te_scores[te_rl].numpy(), bins=60, alpha=0.6, density=True, label="defect regions")
        for t in thresholds:
            plt.axvline(float(t), color="k", ls="--", lw=1)
        plt.xlabel(f"suspiciousness score ({best_name})"); plt.ylabel("density")
        plt.title("Test regions: normal vs defective"); plt.legend(); plt.tight_layout()
        plt.savefig("score_histogram.png", dpi=150)
        print("Saved score_histogram.png")
    except ImportError:
        print("(matplotlib not installed - run 'pip install matplotlib' for the histogram)")
