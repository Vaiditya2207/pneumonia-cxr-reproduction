"""Gradio demo: classify a pediatric chest X-ray as NORMAL / PNEUMONIA and show where the model looked.

Run from the repo root:
    .venv/bin/python demo/train_model.py   # once, writes demo/head.pt
    .venv/bin/python demo/app.py           # opens http://127.0.0.1:7860
"""
import json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
import gradio as gr
from matplotlib import colormaps
from PIL import Image

from train_model import TFM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "demo")
RES = os.path.join(ROOT, "results")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ---------- model: frozen ImageNet InceptionV3 + the trained linear layer ----------
ckpt = torch.load(os.path.join(HERE, "head.pt"))
CLASSES = ckpt["classes"]
backbone = torchvision.models.inception_v3(weights="IMAGENET1K_V1")
backbone.fc = nn.Identity()
backbone = backbone.eval().to(device)
head = nn.Linear(2048, 2); head.load_state_dict(ckpt["state_dict"]); head = head.eval().to(device)

_maps = {}
backbone.Mixed_7c.register_forward_hook(lambda m, i, o: _maps.__setitem__("a", o))


@torch.no_grad()
def predict(img):
    if img is None:
        return None, None, ""
    gray = img.convert("L")
    x = TFM(gray).unsqueeze(0).to(device)
    logits = head(backbone(x))
    prob = F.softmax(logits, 1)[0].cpu().numpy()
    k = int(prob.argmax())

    # Class activation map: the last block's 8x8 feature maps weighted by the linear layer.
    # With global average pooling + one linear layer this is exactly what drives the logit.
    a = _maps["a"][0]                                   # (2048, 8, 8)
    w = head.weight[k] - head.weight[1 - k]             # evidence for the predicted class vs the other
    cam = F.relu(torch.einsum("c,chw->hw", w, a))
    cam = (cam / (cam.max() + 1e-8)).cpu().numpy()
    base = np.asarray(gray.resize((299, 299)), dtype=np.float32) / 255.0
    cam = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize((299, 299), Image.BILINEAR)) / 255.0
    heat = colormaps["jet"](cam)[..., :3]
    overlay = (0.6 * np.stack([base] * 3, -1) + 0.4 * heat) * 255
    overlay = Image.fromarray(np.uint8(overlay.clip(0, 255)))

    note = (f"**Prediction: {CLASSES[k]}** ({100 * prob[k]:.1f}% confident)\n\n"
            "Heatmap = image regions that pushed the model towards this answer (red = strongest).")
    if CLASSES[k] == "PNEUMONIA" and prob[k] < 0.9:
        note += ("\n\nLow-confidence pneumonia call. The model's main weakness is calling NORMAL "
                 "X-rays PNEUMONIA (specificity about 50% on the test set).")
    return {c: float(p) for c, p in zip(CLASSES, prob)}, overlay, note


# ---------- results tab ----------
def mean_std(runs, k):
    v = np.array([r[k] for r in runs])
    return f"{v.mean():.2f}" if len(v) == 1 else f"{v.mean():.2f} ± {v.std():.2f}"


def results_table():
    rows = ["| Model | Seeds | Accuracy | Sensitivity | Specificity | AUC |", "|---|---|---|---|---|---|"]
    keys = ["accuracy", "sensitivity", "specificity", "auc"]
    base = json.load(open(os.path.join(RES, "baseline_metrics.json")))
    rows.append("| Paper (Kermany et al., 2018) | – | " + " | ".join(f"{base['paper'][k]:.1f}" for k in keys) + " |")
    abl = json.load(open(os.path.join(RES, "ablation_all.json")))
    abl["0"] = base["runs"]                             # same Milestone 2 numbers as the README and paper
    names = {"0": "Our reproduction: frozen InceptionV3", "1": "+ 1 block fine-tuned",
             "2": "**+ 2 blocks fine-tuned (our change)**", "all": "Whole network (smaller budget)"}
    for key, name in names.items():
        runs = abl.get(key, [])
        if runs:
            rows.append(f"| {name} | {len(runs)} | " + " | ".join(mean_std(runs, k) for k in keys) + " |")
    return "\n".join(rows)


def fig(name):
    p = os.path.join(RES, name)
    return p if os.path.exists(p) else None


examples = sorted(os.path.join(HERE, "examples", f) for f in os.listdir(os.path.join(HERE, "examples")))
m = ckpt["test_metrics"]

with gr.Blocks(title="Pneumonia CXR demo") as app:
    gr.Markdown("# Pneumonia vs Normal on pediatric chest X-rays\n"
                "Reproduction of Kermany *et al.* (Cell 2018): ImageNet InceptionV3 as a frozen feature "
                "extractor + one trained linear layer. Vaiditya Tanwar & Yashi Gupta.  \n"
                "*Research exercise only, not for clinical use.*")
    with gr.Tab("Try it"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Chest X-ray", height=380)
                btn = gr.Button("Classify", variant="primary")
                gr.Examples(examples, inputs=inp, label="Test-set examples (the last one is a NORMAL the model gets wrong)")
            with gr.Column():
                label = gr.Label(num_top_classes=2, label="Probabilities")
                cam = gr.Image(label="Where the model looked", height=300)
                note = gr.Markdown()
        btn.click(predict, inp, [label, cam, note])
        inp.change(predict, inp, [label, cam, note])
        gr.Markdown(f"This model on the 624-image test set: accuracy {m['accuracy']:.1f}%, "
                    f"sensitivity {m['sensitivity']:.1f}%, specificity {m['specificity']:.1f}%, AUC {m['auc']:.1f}%.")
    with gr.Tab("Results"):
        gr.Markdown("## Official test set (624 images)\n" + results_table())
        gr.Markdown("**Hypothesis** (fine-tuning the top 2 Inception blocks lifts AUC by 1–2 points and fixes "
                    "specificity): **not supported.** Fine-tuned models reach 100% training accuracy without "
                    "improving the test set, which points to a train/test shift rather than weak ImageNet features.")
        with gr.Row():
            gr.Image(fig("confusion_roc.png"), label="Baseline confusion matrix + ROC")
            gr.Image(fig("ablation_bars.png"), label="Ablation: blocks fine-tuned")
        with gr.Row():
            gr.Image(fig("training_curve.png"), label="Training curve (validation ~99%, test ~80%)")
        gr.Image(fig("failures.png"), label="Baseline mistakes on the test set")

if __name__ == "__main__":
    app.launch()
