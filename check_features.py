"""
VisionScout - check that features.pt exists and looks correct.
Run from the project root:  python check_features.py
"""
import os
import torch

FILE = "features.pt"
assert os.path.exists(FILE), f"{FILE} not found - run precompute_features.py first"
print(f"Found {FILE} ({os.path.getsize(FILE) / 1e6:.0f} MB)")

data = torch.load(FILE, weights_only=False)
assert set(data) >= {"train", "test", "grid"}, f"missing top-level keys: {list(data)}"
G = data["grid"]
print("Grid:", G, "x", G)

all_ok = True
for split in ["train", "test"]:
    d = data[split]
    feats, rl, il = d["feats"], d["region_labels"], d["image_labels"]
    n = feats.shape[0]
    checks = {
        "feats shape [N, 49, 512]": tuple(feats.shape[1:]) == (G * G, 512),
        "region_labels shape [N, 49]": tuple(rl.shape) == (n, G * G),
        "image_labels length N": il.shape[0] == n,
        "defect_type/paths length N": len(d["defect_type"]) == n and len(d["paths"]) == n,
        "no NaN/Inf in feats": bool(torch.isfinite(feats.float()).all()),
        "features not all zero": float(feats.float().abs().mean()) > 0,
        "MT_Free tiles have no defect regions":
            all(il[i] == 0 for i, t in enumerate(d["defect_type"]) if t == "MT_Free"),
        "defect tiles have >=1 defect region":
            all(il[i] == 1 for i, t in enumerate(d["defect_type"]) if t != "MT_Free"),
    }
    print(f"\n{split.upper()}: {n} images, {int(il.sum())} defective, "
          f"avg defect regions per defective tile = {rl[il == 1].sum(1).float().mean():.1f}")
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
        all_ok &= ok

print("\nAll checks passed!" if all_ok else "\nSome checks FAILED - paste this output in the chat.")
