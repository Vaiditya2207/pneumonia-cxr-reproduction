# Pneumonia vs Normal on Pediatric Chest X-rays: Reproduction + One Change

Computer Vision course project (research reproduction).

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

## Run it
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# download ChestXRay2017.zip from the Mendeley link above and unzip it into data/ (gives data/chest_xray/{train,test})
.venv/bin/jupyter nbconvert --to notebook --execute --inplace pneumonia_baseline_reproduction.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace pneumonia_milestone4_finetune.ipynb
```
Works on CUDA, Apple-silicon GPU (MPS) or CPU. The baseline takes about 2 minutes on an Apple M5 Pro; Milestone 4 takes about 50 minutes.

*Research exercise only; no clinical claims.*
