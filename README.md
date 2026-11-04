# Pneumonia vs Normal on Pediatric Chest X-rays: Reproduction + One Change

Computer Vision course project (research reproduction) by **Vaiditya Tanwar** and **Yashi Gupta**.

- **Base paper:** D. S. Kermany *et al.*, "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning", *Cell* 172(5), 2018. doi:[10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)
- **Dataset + original code:** [Mendeley Data rscbjbr9sj v2](https://data.mendeley.com/datasets/rscbjbr9sj/2) (`ChestXRay2017.zip`, `code2017.zip`)
- **Baseline (paper's method):** ImageNet InceptionV3, fully frozen, plus a new linear softmax layer (Adam, lr 1e-3, batch 256, 4000 steps, 299×299 input).
- **Our one change:** fine-tune the last two Inception blocks (`Mixed_7b`, `Mixed_7c`) at lr 1e-4. Ablation over 0 / 1 / 2 blocks and the whole network.

## Contents
| Milestone | File |
|---|---|
| M2: baseline reproduced | `pneumonia_baseline_reproduction.ipynb` |
| M3: hypothesis (written before coding) + introduction + slides | `milestone3/` |
| M4: implementation + ablation | `pneumonia_milestone4_finetune.ipynb` |
| M5: IEEE-format paper + defence Q&A | `milestone5/paper.pdf`, `milestone5/defence_qa.md` |
| Figures and metrics | `results/*.png`, `results/*.json` |

## Results (official test set, 624 images)
| Model | Seeds | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|---|
| Paper (Kermany et al.) | – | 92.8 | 93.2 | 90.1 | 96.8 |
| Our reproduction: frozen InceptionV3 | 5 | 80.1 ± 0.7 | 98.4 ± 0.3 | 49.7 ± 1.9 | 92.9 ± 0.2 |
| + 1 block fine-tuned | 1 | 82.2 | 99.7 | 53.0 | 92.3 |
| **+ 2 blocks fine-tuned (our change)** | 2 | 80.9 ± 0.4 | 99.5 ± 0.3 | 49.8 ± 1.5 | 92.6 ± 0.9 |
| whole network (smaller budget) | 1 | 78.4 | 100.0 | 42.3 | 93.9 |

- **Reproduction:** the AUC is within 4 points of the paper, but specificity is much lower. Almost all errors are NORMAL X-rays called PNEUMONIA. Validation accuracy is about 98.7% but test accuracy is about 80%, which points to a train/test distribution shift.
- **Hypothesis (pre-registered in `milestone3/hypothesis.md`):** fine-tuning the top 2 blocks would raise AUC by 1–2 points and fix specificity. **Not supported:** AUC −0.27, specificity +0.13. Fine-tuned models reach 100% training accuracy without improving the test set, so the bottleneck is the train/test shift and class imbalance, not the ImageNet features.

## Run it
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# download ChestXRay2017.zip from the Mendeley link above and unzip it into data/ (gives data/chest_xray/{train,test})
.venv/bin/jupyter nbconvert --to notebook --execute --inplace pneumonia_baseline_reproduction.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace pneumonia_milestone4_finetune.ipynb
```
Works on CUDA, Apple-silicon GPU (MPS) or CPU. The baseline takes about 2 minutes on an Apple M5 Pro; Milestone 4 takes about 50 minutes.

*Research exercise only; no clinical claims.*
