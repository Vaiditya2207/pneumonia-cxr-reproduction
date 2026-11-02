# Hypothesis: Fine-tune the top of InceptionV3

## The one change
The base paper (Kermany et al., 2018) keeps the ImageNet-pretrained InceptionV3 **fully frozen** and trains only a new linear softmax layer on its 2048-d features.

**Our change:** unfreeze the **last two Inception blocks (Mixed_7b, Mixed_7c)** and train them together with the new head at a **small learning rate (1e-4)**. Everything else stays identical: same data split, 299×299 input, preprocessing, optimiser (Adam), batch size and number of training steps.

## Prediction (written before running the experiment)
Compared with our reproduced frozen baseline (Acc 80.1%, Sens 98.4%, Spec 49.7%, AUC 92.9%):

- **AUC increases by about 1–2 percentage points.**
- **Specificity improves the most.** We expect most baseline errors to be NORMAL images predicted as PNEUMONIA (the training set is roughly 3:1 pneumonia:normal), and better X-ray-specific features should reduce these false positives.
- Sensitivity stays about the same (within ~1 point).

## Reasoning
- ImageNet features are learned from natural **colour photographs**; chest X-rays are **grayscale** images with very different structure.
- Early layers learn generic edges and textures that transfer well; the top layers are the most specific to ImageNet objects (Yosinski et al., 2014). So the top layers are the ones most worth adapting.
- Unfreezing **only** the last two blocks lets the model learn X-ray patterns such as opacities and consolidation, while the frozen lower layers and the small learning rate limit overfitting on only ~5k training images.

## How we will test it
1. Train the baseline (0 blocks unfrozen) and the modified model (2 blocks unfrozen) with the same settings and seed.
2. Evaluate both on the same official test set (624 images: 234 NORMAL, 390 PNEUMONIA).
3. Report accuracy, sensitivity, specificity and ROC-AUC, plus the confusion matrix, to see *which* errors changed.
4. Milestone 4 ablation: unfreeze **0 / 1 / 2 blocks / whole network** and compare the same four metrics.

## What would prove it wrong
The hypothesis is **falsified** if the fine-tuned model:
- does **not** improve AUC over the baseline (gain ≤ 0, or within run-to-run noise across seeds), **or**
- improves AUC but specificity does **not** improve (the gain comes only from sensitivity), **or**
- trains to much better accuracy on the training set but scores lower on the test set (overfitting / train–test shift cancels the gain).

## Risks and limitations
- Small dataset → fine-tuning may overfit; the test set comes from the same hospital, so results may not generalise.
- A single test set and few runs give limited statistical certainty.
- This is a research reproduction only; no clinical claims are made.
