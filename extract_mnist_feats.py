"""Extract 512-D deep features for MNIST via a pretrained ResNet18
(penultimate layer) -> mnist_feats.npz. This is the 'MNIST features' dataset
from the proposal (deep embeddings, not raw pixels).

Uses the EXACT same 10k samples (and order) as the raw-pixel runs — X/y are
taken from idc_out_mnist_seed0.npz — so all comparisons stay sample-identical
to the raw-pixel MNIST rows."""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
from wpaths import DATA, idc_out

dev = "cuda" if torch.cuda.is_available() else "cpu"

d = np.load(idc_out("idc_out_mnist_seed0.npz"))
X_px = np.ascontiguousarray(d["X"], np.float32)      # (10000, 784) in [0,1]
y = d["y_true"].astype(int)

net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
net.fc = torch.nn.Identity()            # 512-D penultimate features
net.eval().to(dev)

mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

feats = []
with torch.no_grad():
    for s in range(0, len(X_px), 256):
        xb = torch.tensor(X_px[s:s + 256], device=dev).view(-1, 1, 28, 28)
        xb = xb.repeat(1, 3, 1, 1)                       # grayscale -> 3 channels
        xb = F.interpolate(xb, size=224, mode="bilinear", align_corners=False)
        xb = (xb - mean) / std
        feats.append(net(xb).cpu().numpy())
X = np.concatenate(feats).astype(np.float32)
np.savez(os.path.join(DATA, "mnist_feats.npz"), X=X, y=y)
print("mnist feats:", X.shape, "classes:", np.unique(y), "counts:", np.bincount(y))
