"""Extract 512-D deep features from a 10k CIFAR-10 subsample via a pretrained
ResNet18 (penultimate layer) -> cifar_feats.npz. This is the 'CIFAR-10 features'
dataset from the proposal (deep embeddings, not raw pixels)."""
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms, models

HERE = os.path.dirname(os.path.abspath(__file__))
dev = "cuda" if torch.cuda.is_available() else "cpu"

tf = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
from wpaths import DATA
ds = torchvision.datasets.CIFAR10(os.path.join(DATA, "cifar_data"), train=True,
                                  download=True, transform=tf)
idx = np.random.RandomState(0).permutation(len(ds))[:10000]

net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
net.fc = torch.nn.Identity()            # 512-D penultimate features
net.eval().to(dev)

loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx),
                                     batch_size=128, num_workers=0)
feats, ys = [], []
with torch.no_grad():
    for xb, yb in loader:
        feats.append(net(xb.to(dev)).cpu().numpy())
        ys.append(yb.numpy())
X = np.concatenate(feats).astype(np.float32)
y = np.concatenate(ys).astype(int)
np.savez(os.path.join(DATA, "cifar_feats.npz"), X=X, y=y)
print("cifar feats:", X.shape, "classes:", np.unique(y), "counts:", np.bincount(y))
