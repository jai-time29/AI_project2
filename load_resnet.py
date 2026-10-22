"""
VisionScout - Step 2: Load a pretrained ResNet18 and extract region features.

Run from the root of your project repo:
    python load_resnet.py
"""
import glob
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

DATA_ROOT = "data/MagneticTile"   # where the dataset was cloned
GRID = 7                          # 7x7 = 49 regions (matches ResNet's output map)

# ---- 1. Use GPU if available, otherwise CPU ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---- 2. Load ResNet18 with ImageNet-pretrained weights ----
# The first run downloads the weights (~45 MB) automatically.
weights = models.ResNet18_Weights.DEFAULT
resnet = models.resnet18(weights=weights)

# ---- 3. Remove the last two layers (avgpool + fc classifier) ----
# We don't want ImageNet class predictions; we want the 7x7 spatial feature map.
backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
backbone.eval().to(device)            # eval mode: no dropout / batchnorm updates
for p in backbone.parameters():
    p.requires_grad = False           # freeze: we only use it as a feature extractor

# ---- 4. Preprocessing (same normalization ResNet was trained with) ----
img_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
mask_tf = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
    T.ToTensor(),
])

# ---- 5. Pick one defective image and its mask ----
img_paths = sorted(glob.glob(f"{DATA_ROOT}/MT_Blowhole/Imgs/*.jpg"))
print("Found", len(img_paths), "Blowhole images")
img_path = img_paths[0]
mask_path = img_path.replace(".jpg", ".png")

img = img_tf(Image.open(img_path).convert("RGB"))   # grayscale -> 3 channels
mask = mask_tf(Image.open(mask_path).convert("L"))

# ---- 6. Extract features ----
with torch.no_grad():
    fmap = backbone(img.unsqueeze(0).to(device))     # [1, 512, 7, 7]
region_feats = fmap.flatten(2).squeeze(0).T.cpu()   # [49, 512]: one vector per region

# ---- 7. Region labels from the mask (1 = region contains a defect) ----
region_labels = F.adaptive_max_pool2d(mask, GRID).flatten()   # [49]
region_labels = (region_labels > 0).float()

print("Image:", img_path)
print("Feature map shape:", tuple(fmap.shape))
print("Region features shape:", tuple(region_feats.shape))
print("Defective regions:", int(region_labels.sum()), "of", GRID * GRID)
print("Defect grid (1 = defect):")
print(region_labels.view(GRID, GRID).int())
