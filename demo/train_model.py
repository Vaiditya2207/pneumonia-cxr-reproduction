"""Train the baseline model (frozen ImageNet InceptionV3 + linear layer) and save it for the demo.

Same recipe as pneumonia_baseline_reproduction.ipynb, seed 0: Adam lr 1e-3, batch 256, 4000 steps,
90/10 train/val split. Reuses results/feats_*.pt if the notebook already cached them,
otherwise extracts features from data/chest_xray (about 2 min on Apple silicon).

Run from the repo root:  .venv/bin/python demo/train_model.py
Writes demo/head.pt (about 20 KB) and copies a few test X-rays into demo/examples/.
"""
import os, random, shutil
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, confusion_matrix
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "chest_xray")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "demo")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

TFM = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def features(split):
    cache = os.path.join(RES, f"feats_{split}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    net = torchvision.models.inception_v3(weights="IMAGENET1K_V1")
    net.fc = nn.Identity(); net = net.eval().to(device)
    ds = datasets.ImageFolder(os.path.join(DATA, split), transform=TFM)
    f, y = [], []
    for x, t in torch.utils.data.DataLoader(ds, batch_size=64, num_workers=4):
        f.append(net(x.to(device)).cpu()); y.append(t)
    out = (torch.cat(f), torch.cat(y))
    os.makedirs(RES, exist_ok=True); torch.save(out, cache)
    return out


def train_head(f_train, y_train, seed=0, steps=4000, batch=256, lr=1e-3):
    set_seed(seed)
    perm = torch.randperm(len(f_train), generator=torch.Generator().manual_seed(seed))
    tr = perm[int(0.1 * len(f_train)):]
    X, Y = f_train[tr].to(device), y_train[tr].to(device)
    head = nn.Linear(2048, 2).to(device)
    nn.init.normal_(head.weight, std=0.001); nn.init.zeros_(head.bias)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (batch,), device=device)
        loss = F.cross_entropy(head(X[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return head.cpu()


def main():
    f_train, y_train = features("train")
    f_test, y_test = features("test")
    head = train_head(f_train, y_train)

    with torch.no_grad():
        prob = F.softmax(head(f_test), 1)[:, 1].numpy()
    y = y_test.numpy(); pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    m = {"accuracy": 100 * (tp + tn) / len(y), "sensitivity": 100 * tp / (tp + fn),
         "specificity": 100 * tn / (tn + fp), "auc": 100 * roc_auc_score(y, prob)}
    m = {k: float(v) for k, v in m.items()}            # plain floats so torch.load(weights_only) accepts them
    print("test set: " + "  ".join(f"{k} {v:.1f}" for k, v in m.items()))
    torch.save({"state_dict": head.state_dict(), "classes": ["NORMAL", "PNEUMONIA"], "test_metrics": m},
               os.path.join(OUT, "head.pt"))

    # Example images: 2 correct NORMAL, 2 correct PNEUMONIA, 1 NORMAL the model gets wrong.
    # ImageFolder sorts files, so the cached test features line up with this listing.
    paths = [p for p, _ in datasets.ImageFolder(os.path.join(DATA, "test")).samples]
    picks = {"normal_1": np.where((y == 0) & (pred == 0))[0][0],
             "normal_2": np.where((y == 0) & (pred == 0))[0][7],
             "pneumonia_1": np.where((y == 1) & (pred == 1))[0][0],
             "pneumonia_2": np.where((y == 1) & (pred == 1))[0][11],
             "normal_misclassified": np.where((y == 0) & (pred == 1))[0][0]}
    ex_dir = os.path.join(OUT, "examples")
    shutil.rmtree(ex_dir, ignore_errors=True); os.makedirs(ex_dir)
    for name, i in picks.items():
        img = Image.open(paths[i]).convert("L"); img.thumbnail((640, 640))
        img.save(os.path.join(ex_dir, f"{name}.jpg"), quality=88)
        print(f"example {name}: P(pneumonia) = {prob[i]:.2f}  ({os.path.basename(paths[i])})")


if __name__ == "__main__":
    main()
