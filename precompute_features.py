"""
VisionScout - Step 3: Run ResNet18 on every image once and save the features.

Output file: features.pt  (about 70 MB, do NOT commit it to GitHub)

Run from the root of your project repo:
    python precompute_features.py
"""
import glob
import os
import random

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

DATA_ROOT = "data/MagneticTile"
OUT_FILE = "features.pt"
GRID = 7            # 7x7 = 49 regions
TEST_FRAC = 0.2     # 20% of each folder goes to the test set
BATCH_SIZE = 32
SEED = 42

img_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
mask_tf = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
    T.ToTensor(),
])


def collect_samples():
    """Find every (image, mask, folder) and split each folder 80/20."""
    rng = random.Random(SEED)
    train, test = [], []
    for folder in sorted(glob.glob(f"{DATA_ROOT}/MT_*")):
        name = os.path.basename(folder)               # e.g. "MT_Crack"
        samples = []
        for img_path in sorted(glob.glob(f"{folder}/Imgs/*.jpg")):
            mask_path = img_path.replace(".jpg", ".png")
            mask = Image.open(mask_path).convert("L")
            # Skip defect images whose mask is completely black (4 in MT_Uneven)
            if name != "MT_Free" and mask.getextrema()[1] == 0:
                continue
            samples.append((img_path, mask_path, name))
        rng.shuffle(samples)
        n_test = int(len(samples) * TEST_FRAC)
        test += samples[:n_test]
        train += samples[n_test:]
        print(f"{name:12s} total={len(samples):4d}  train={len(samples) - n_test:4d}  test={n_test:4d}")
    return train, test


class TileDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, mask_path, _ = self.samples[i]
        img = img_tf(Image.open(img_path).convert("RGB"))       # grayscale -> 3 channels
        mask = mask_tf(Image.open(mask_path).convert("L"))
        region_labels = (F.adaptive_max_pool2d(mask, GRID).flatten() > 0).float()   # [49]
        return img, region_labels


def load_backbone(device):
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-2])   # keep 7x7 feature map
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def extract(samples, backbone, device):
    loader = torch.utils.data.DataLoader(TileDataset(samples), batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)
    all_feats, all_labels = [], []
    with torch.no_grad():
        for b, (imgs, labels) in enumerate(loader):
            fmap = backbone(imgs.to(device))                  # [B, 512, 7, 7]
            feats = fmap.flatten(2).transpose(1, 2)           # [B, 49, 512]
            all_feats.append(feats.half().cpu())              # float16 halves the file size
            all_labels.append(labels)
            print(f"  batch {b + 1}/{len(loader)}", end="\r")
    print()
    return torch.cat(all_feats), torch.cat(all_labels)


def build_split(samples, backbone, device):
    feats, region_labels = extract(samples, backbone, device)
    return {
        "feats": feats,                                        # [N, 49, 512]
        "region_labels": region_labels,                        # [N, 49]  1 = defect in region
        "image_labels": (region_labels.sum(1) > 0).long(),     # [N]      1 = defective tile
        "defect_type": [s[2] for s in samples],                # folder name per image
        "paths": [s[0] for s in samples],                      # for visualising later
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_samples, test_samples = collect_samples()
    backbone = load_backbone(device)

    print("Extracting train features...")
    train = build_split(train_samples, backbone, device)
    print("Extracting test features...")
    test = build_split(test_samples, backbone, device)

    torch.save({"train": train, "test": test, "grid": GRID}, OUT_FILE)

    for name, split in [("train", train), ("test", test)]:
        print(f"{name}: feats {tuple(split['feats'].shape)}, "
              f"defective tiles {int(split['image_labels'].sum())}/{len(split['image_labels'])}")
    print(f"Saved to {OUT_FILE} ({os.path.getsize(OUT_FILE) / 1e6:.0f} MB)")
