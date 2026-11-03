"""Generate milestone3/slides.pptx. Fill the baseline numbers below and re-run:
    .venv/bin/python milestone3/make_slides.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

# ---- Our reproduction results (fill these in) ----
ACC = "80.1%"
SENS = "98.4%"
SPEC = "49.7%"
AUC = "92.9%"

# ---- Paper numbers (confirm against Kermany et al. 2018, Fig. 6) ----
PAPER = {"Accuracy": "92.8%", "Sensitivity": "93.2%", "Specificity": "90.1%", "AUC": "96.8%"}

OUT = Path(__file__).with_name("slides.pptx")
ROOT = Path(__file__).resolve().parent.parent
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
TEAL = RGBColor(0x2A, 0x9D, 0x8F)
GREY = RGBColor(0x44, 0x44, 0x44)
LIGHT = RGBColor(0xE8, 0xF1, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Arial"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, lines, size=24, color=GREY, bold=False, align=PP_ALIGN.LEFT, bullets=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate([lines] if isinstance(lines, str) else lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("•  " + line) if bullets else line
        p.alignment = align
        p.space_after = Pt(12)
        f = p.font
        f.size, f.color.rgb, f.bold, f.name = Pt(size), color, bold, FONT
    return tb


def new_slide(title):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.2))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    text(s, 0.6, 0.22, 12, 0.8, title, size=34, color=WHITE, bold=True)
    return s


def box(slide, x, y, w, h, label, fill=LIGHT, color=NAVY, size=18):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = fill; shp.line.color.rgb = NAVY
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.text = label
    for p in tf.paragraphs:
        p.alignment = PP_ALIGN.CENTER
        p.font.size, p.font.bold, p.font.color.rgb, p.font.name = Pt(size), True, color, FONT
    return shp


def arrow(slide, x1, y, x2):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y), Inches(x2), Inches(y))
    c.line.color.rgb = NAVY; c.line.width = Pt(2.5)
    c.line._get_or_add_ln().append(
        c.line._get_or_add_ln().makeelement(
            "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle"}))


def pipeline(slide, y, frozen_label, frozen_fill, frozen_size=18):
    xs, w, h = [0.4, 2.95, 5.85, 8.3, 10.75], [2.1, 2.5, 2.0, 2.0, 2.2], 1.4
    labels = ["Chest X-ray\n299×299", frozen_label, "2048\nfeatures", "Linear\nlayer", "Normal /\nPneumonia"]
    fills = [LIGHT, frozen_fill, LIGHT, RGBColor(0xD6, 0xF0, 0xEC), LIGHT]
    for i, (x, ww, lab, fl) in enumerate(zip(xs, w, labels, fills)):
        box(slide, x, y, ww, h, lab, fill=fl, size=frozen_size if i == 1 else 18)
    for i in range(4):
        arrow(slide, xs[i] + w[i], y + h / 2, xs[i + 1])


# 1. Title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
text(s, 0.8, 2.2, 11.7, 1.8, "Pneumonia vs Normal on Pediatric Chest X-rays",
     size=44, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, 0.8, 4.0, 11.7, 1.2, "Reproducing Kermany et al. (Cell, 2018) + one change",
     size=26, color=RGBColor(0xCF, 0xE3, 0xF2), align=PP_ALIGN.CENTER)
text(s, 0.8, 5.4, 11.7, 0.8, "Milestones 2–5", size=20, color=RGBColor(0xCF, 0xE3, 0xF2), align=PP_ALIGN.CENTER)

# 2. Introduction
s = new_slide("Why this problem?")
text(s, 0.8, 1.7, 11.8, 5.2, [
    "Pneumonia: 740 180 deaths in children under 5 in 2019 (14% of under-5 deaths) — WHO",
    "Chest X-rays are the main test, but subtle and hard to read",
    "Radiologists are scarce where the burden is highest",
    "Deep learning + transfer learning (ImageNet → X-rays) can help as a support tool",
], size=26, bullets=True)

# 3. Base paper & method
s = new_slide("Base paper & method")
text(s, 0.6, 1.45, 12.2, 0.9,
     "Kermany et al., Cell 172(5), 2018 · ~5.2k train / 624 test images (Normal vs Pneumonia)", size=20)
pipeline(s, 2.8, "InceptionV3\n(ImageNet)\nFROZEN", RGBColor(0xDD, 0xDD, 0xDD))
text(s, 0.6, 5.0, 12.2, 1.8, [
    "Only the new linear layer is trained",
    "Adam, lr 0.001 · batch 256 · 4000 steps",
], size=22, bullets=True)

# 4. Paper vs reproduction
s = new_slide("Paper vs our reproduction")
ours = {"Accuracy": ACC, "Sensitivity": SENS, "Specificity": SPEC, "AUC": AUC}
rows, cols = 5, 3
tbl = s.shapes.add_table(rows, cols, Inches(2.4), Inches(1.8), Inches(8.5), Inches(4.0)).table
for c, h in enumerate(["Metric", "Paper", "Ours (frozen)"]):
    tbl.cell(0, c).text = h
for r, m in enumerate(PAPER, start=1):
    tbl.cell(r, 0).text, tbl.cell(r, 1).text, tbl.cell(r, 2).text = m, PAPER[m], ours[m]
for r in range(rows):
    for c in range(cols):
        cell = tbl.cell(r, c)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            p.font.size, p.font.bold, p.font.name = Pt(24), r == 0, FONT
            p.font.color.rgb = WHITE if r == 0 else GREY
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY if r == 0 else (LIGHT if r % 2 else WHITE)
text(s, 2.4, 6.1, 8.5, 0.6, "Test set: 624 images (234 Normal / 390 Pneumonia). Ours: mean of 5 seeds. Paper values: confirm against Fig. 6.",
     size=16, align=PP_ALIGN.CENTER)

# 4b. Where the baseline fails
s = new_slide("Where the baseline fails")
s.shapes.add_picture(str(ROOT / "results" / "confusion_roc.png"), Inches(0.4), Inches(1.45), width=Inches(7.9))
text(s, 8.5, 1.6, 4.5, 5.6, [
    "119 of 127 errors: NORMAL called PNEUMONIA",
    "Val acc 98.7% vs test acc 80% → train/test shift",
    "3:1 class imbalance pushes toward 'pneumonia'",
    "AUC within ~4 pts of paper; gap is mostly specificity",
], size=19, bullets=True)

# 5. Hypothesis
s = new_slide("Hypothesis: the one change")
pipeline(s, 1.9, "InceptionV3\n+ Mixed_7b/7c\nTRAINED", RGBColor(0xFF, 0xE8, 0xC2), frozen_size=16)
text(s, 0.6, 3.9, 12.2, 3.0, [
    "Unfreeze only the last 2 Inception blocks (Mixed_7b, Mixed_7c)",
    "Train them with the head at a small learning rate (1e-4)",
    "Everything else identical to the baseline",
], size=24, bullets=True)

# 6. Prediction & reasoning
s = new_slide("Prediction & reasoning")
text(s, 0.6, 1.5, 6.0, 0.6, "Prediction", size=26, color=TEAL, bold=True)
text(s, 0.6, 2.2, 6.0, 4.6, [
    "AUC up ~1–2 points",
    "Specificity improves most (fewer Normal → Pneumonia errors)",
    "Wrong if: no AUC gain, or gain without better specificity",
], size=22, bullets=True)
text(s, 6.9, 1.5, 6.0, 0.6, "Reasoning", size=26, color=TEAL, bold=True)
text(s, 6.9, 2.2, 6.0, 4.6, [
    "ImageNet = colour photos; X-rays = grayscale",
    "Top layers learn X-ray patterns (opacities)",
    "Early layers stay frozen → less overfitting on ~5k images",
    "Risk: overfitting / train–test shift",
], size=22, bullets=True)

# 7. Plan / ablation
s = new_slide("Plan: Milestone 4 ablation")
for i, lab in enumerate(["0 blocks\n(baseline)", "1 block", "2 blocks\n(ours)", "Whole\nnetwork"]):
    box(s, 0.9 + i * 3.0, 1.9, 2.5, 1.5, lab, fill=RGBColor(0xFF, 0xE8, 0xC2) if i == 2 else LIGHT, size=22)
text(s, 0.8, 3.9, 11.8, 3.0, [
    "Report accuracy, sensitivity, specificity, AUC + confusion matrix",
    "Same data, seed and settings for every run",
    "Limitations: small single-hospital dataset; no clinical claims",
], size=24, bullets=True)

# ---------- Milestone 4 & 5 slides (generated from ../results/ablation_all.json) ----------
import json
import numpy as np
ABL = ROOT / "results" / "ablation_all.json"
if ABL.exists():
    R = json.load(open(ABL))
    NAMES = {"0": "0 blocks (baseline)", "1": "1 block", "2": "2 blocks (ours)", "all": "Whole network*"}
    mean = lambda k, m: float(np.mean([r[m] for r in R[k]]))

    s = new_slide("Milestone 4: ablation results")
    keys = [k for k in ["0", "1", "2", "all"] if k in R]
    cols = ["Unfrozen", "Seeds", "Acc", "Sens", "Spec", "AUC", "Train acc"]
    tbl = s.shapes.add_table(len(keys) + 1, len(cols), Inches(0.5), Inches(1.5), Inches(12.3), Inches(3.2)).table
    for c, h in enumerate(cols):
        tbl.cell(0, c).text = h
    for r, k in enumerate(keys, start=1):
        vals = [NAMES[k], str(len(R[k]))] + [f"{mean(k, m):.1f}" for m in ["accuracy", "sensitivity", "specificity", "auc", "train_acc"]]
        for c, v in enumerate(vals):
            tbl.cell(r, c).text = v
    for r in range(len(keys) + 1):
        for c in range(len(cols)):
            cell = tbl.cell(r, c); cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                p.font.size, p.font.bold, p.font.name = Pt(20), r == 0 or keys[r - 1] == "2" if r else True, FONT
                p.font.color.rgb = WHITE if r == 0 else GREY
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if r == 0 else (RGBColor(0xFF, 0xE8, 0xC2) if keys[r - 1] == "2" else (LIGHT if r % 2 else WHITE))
    text(s, 0.5, 5.0, 12.3, 1.8, [
        "Same data, split, batch 256, 4000 steps; only the unfrozen part changes (blocks lr 1e-4, head lr 1e-3)",
        "*Whole network: smaller budget (batch 32, 1000 steps, 1 seed) — rough guide only",
        "Test set: 624 images; mean over seeds; threshold 0.5",
    ], size=17, bullets=True)

    s = new_slide("Ablation at a glance")
    s.shapes.add_picture(str(ROOT / "results" / "ablation_bars.png"), Inches(0.5), Inches(1.45), width=Inches(12.3))

    d = {m: mean("2", m) - mean("0", m) for m in ["auc", "specificity", "sensitivity", "accuracy"]}
    auc_std0 = float(np.std([r["auc"] for r in R["0"]]))
    checks = [("AUC up ~1–2 points", d["auc"] > max(0.5, 2 * auc_std0), f"{d['auc']:+.1f} pts"),
              ("Specificity improves the most", d["specificity"] > 0 and d["specificity"] >= max(d["sensitivity"], d["auc"]), f"{d['specificity']:+.1f} pts"),
              ("Sensitivity within ~1 point", abs(d["sensitivity"]) <= 1.5, f"{d['sensitivity']:+.1f} pts")]
    s = new_slide("Was our prediction right?")
    for i, (claim, ok, delta) in enumerate(checks):
        y = 1.7 + i * 1.35
        box(s, 0.8, y, 1.9, 1.0, "YES" if ok else "NO", fill=RGBColor(0xD6, 0xF0, 0xEC) if ok else RGBColor(0xF8, 0xD7, 0xDA), size=24)
        text(s, 3.0, y + 0.18, 6.5, 0.8, claim, size=24)
        text(s, 9.6, y + 0.18, 3.2, 0.8, delta, size=24, bold=True, color=NAVY)
    n_ok = sum(ok for _, ok, _ in checks)
    verdict = "supported" if n_ok == 3 else "partly supported" if checks[0][1] or checks[1][1] else "not supported"
    text(s, 0.8, 5.9, 11.8, 1.0, f"Hypothesis {verdict}: 2 blocks vs frozen baseline, accuracy {d['accuracy']:+.1f} pts. "
         "Prediction was written before the experiment.", size=20, color=TEAL, bold=True)

    s = new_slide("Failure analysis")
    s.shapes.add_picture(str(ROOT / "results" / "failures_finetuned.png"), Inches(0.3), Inches(1.45), width=Inches(12.7))
    r0, r2 = R["0"][0], R["2"][0]
    text(s, 0.6, 4.5, 12.2, 2.8, [
        f"Normal → Pneumonia errors: {r0['fp']} (frozen) vs {r2['fp']} (2 blocks); missed pneumonia: {r0['fn']} vs {r2['fn']}",
        f"Train acc {mean('2', 'train_acc'):.1f}% / val {mean('2', 'val_acc'):.1f}% vs test {mean('2', 'accuracy'):.1f}% → train/test shift dominates",
        "3:1 class imbalance + threshold 0.5 bias the model toward 'pneumonia'",
    ], size=19, bullets=True)

    s = new_slide("Limitations & next steps")
    text(s, 0.6, 1.6, 6.0, 0.6, "Limitations", size=26, color=TEAL, bold=True)
    text(s, 0.6, 2.3, 6.0, 4.6, [
        "One hospital, one fixed 624-image test set",
        "Few seeds for fine-tuned runs",
        "Whole-network row uses a smaller budget",
        "Research exercise only — no clinical claims",
    ], size=21, bullets=True)
    text(s, 6.9, 1.6, 6.0, 0.6, "Next steps", size=26, color=TEAL, bold=True)
    text(s, 6.9, 2.3, 6.0, 4.6, [
        "Class-weighted loss for the 3:1 imbalance",
        "Choose threshold on validation, not 0.5",
        "Data augmentation against train/test shift",
        "External test set (another hospital)",
    ], size=21, bullets=True)

    s = new_slide("Conclusion")
    text(s, 0.8, 1.7, 11.8, 5.0, [
        f"Reproduced Kermany et al.: AUC {AUC} vs 96.8% (paper); low specificity ({SPEC}) is the main gap",
        f"One change (fine-tune top 2 blocks): AUC {mean('2', 'auc'):.1f}%, specificity {mean('2', 'specificity'):.1f}%, accuracy {mean('2', 'accuracy'):.1f}%",
        f"Hypothesis {verdict} against criteria fixed in advance",
        "Code, results and paper: github.com/Vaiditya2207/pneumonia-cxr-reproduction",
    ], size=24, bullets=True)

prs.save(OUT)
print(f"saved {OUT}")
