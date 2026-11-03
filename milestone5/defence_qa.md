# Defence Q&A: Pneumonia CXR Reproduction (Milestone 5)

Short answers to the questions an examiner is most likely to ask. Numbers come from `results/baseline_metrics.json` and the notebooks. Milestone 4 numbers are in `paper.pdf`, Table III and Table IV.

---

**1. Why did you pick this paper?**
Kermany et al. (Cell, 2018) is well cited and the dataset is public with an official train/test split. The method is simple to reproduce: a frozen ImageNet InceptionV3 plus one linear layer, and the authors' training settings are known (Adam, lr 1e-3, batch 256, 4000 steps, 299×299). Training only the head takes seconds, so we could afford 5 seeds. The frozen design also suggests an obvious, testable change.

**2. What exactly did you reproduce?**
The chest X-ray part only, not the retinal OCT part. We used the same official split (train 1349 NORMAL / 3883 PNEUMONIA, test 234 / 390), the same input size and network, and the same head and optimiser settings. We report accuracy, sensitivity, specificity, precision and ROC-AUC as mean ± std over 5 seeds.

**3. Your accuracy is 80% vs 92.8% in the paper. Did the reproduction fail?**
Not fully. AUC, which does not depend on a threshold, is 92.9 vs 96.8, a gap of about 4 points. Most of the accuracy gap comes from the 0.5 threshold. Sensitivity is actually higher than the paper's (98.4 vs 93.2), but specificity is much lower (49.7 vs 90.1). The seed std is under 2 points, so the gap is not random noise. We call it a partial reproduction: same ranking ability within a few AUC points, very different operating point.

**4. Why is specificity so low?**
For seed 0, 119 of the 127 errors are NORMAL images called PNEUMONIA. Three causes:
- **Train/test shift.** Validation accuracy on held-out training images is 98.66%, but test accuracy is 79.65%. The test set was collected and labelled separately, so the model learned cues that do not carry over.
- **Class imbalance.** About 3 pneumonia images per normal one, and no class weighting, push predicted probabilities up.
- **Threshold.** 0.5 is arbitrary. At the paper's sensitivity (93.6%, threshold 0.945), our specificity rises to 70.9%.

**5. What could explain the remaining AUC gap?**
We use torchvision's InceptionV3 weights, not the 2015 TensorFlow graph the authors used. We also resize with PIL instead of TensorFlow and use our own validation split. The paper does not say which threshold or checkpoint it used. Any of these could move AUC by a few points. We could only confirm the 92.8% accuracy from secondary sources; we did not verify the other three paper numbers against the original text (footnote in Table II).

**6. Why that one change: unfreezing the last two Inception blocks?**
ImageNet features come from colour photos of objects; X-rays are grayscale and the useful patterns (opacities, consolidation) are different. Yosinski et al. show the top layers are the most task-specific, and Raghu et al. show the top of the network changes the most when fine-tuning on medical images. Unfreezing only Mixed_7b and Mixed_7c at a 10× smaller learning rate lets the high-level features adapt. It also limits overfitting on about 5k images, and the frozen lower part can be cached so it runs on a laptop.

**7. What did you predict, and was it right?**
Before running the experiment we predicted: AUC +1–2 points, specificity improving the most, and sensitivity within ~1 point. See Table IV in the paper for observed changes and the verdict on each criterion. These are filled by `fill_macros.py` from `results/ablation_all.json`. Only 1–2 seeds were run for the fine-tuned settings, so this result is less certain than the reproduction.

**8. What result would have falsified your hypothesis?**
We wrote three criteria down in advance (milestone3/hypothesis.md):
- the AUC gain is ≤ 0 or within seed noise (we use > 0.5 points and > 2× baseline seed std);
- AUC rises but specificity does not, meaning the gain comes from sensitivity;
- training accuracy rises while test performance does not, meaning overfitting or shift cancels the gain.

**9. How do you know the fine-tuned model is not just overfitting?**
Compare train accuracy with test accuracy in Table III. Near-100% train accuracy with little test gain means extra capacity is fitting the training hospital's images. That is the same train/test shift we see for the baseline. A truly external test set would be the real check, and we do not have one.

**10. Why is the whole-network row not directly comparable?**
Fine-tuning all 22M parameters cannot use the feature cache. Every step needs a full forward and backward pass on images, so 4000 steps at batch 256 would take hours on a laptop GPU. That row uses batch 32, 1000 steps and 1 seed, and is marked with an asterisk.

**11. Is accuracy the right metric here?**
Not on its own. The test set is 62.5% pneumonia, so a model that always says PNEUMONIA already gets 62.5% accuracy. Sensitivity and specificity show which error is made, and AUC measures ranking independent of the threshold. In a screening setting, missed pneumonia (false negatives) and false alarms have different costs. That is why we report all of them and the confusion matrix.

**12. What would you do next?**
Each item is cheap and targets a specific failure we found:
- class-weighted loss or balanced sampling for the 1:3 imbalance;
- choosing the threshold on validation data, ideally a validation set that resembles the test set;
- X-ray-appropriate augmentation (small rotations, shifts, contrast changes; no horizontal flips that break anatomy cues);
- more seeds for the fine-tuned models;
- external validation on a different dataset such as ChestX-ray14 or RSNA pneumonia.

**13. What are the main limitations?**
- One hospital, one fixed 624-image test set, no external validation.
- Few seeds for fine-tuning, and a reduced budget for the whole network.
- Paper numbers only partly verifiable.
- Fixed 0.5 threshold and no calibration.

This is a course reproduction study. The models make no clinical claims and must not be used for diagnosis.

**14. Could the model be using shortcuts instead of lung findings?**
Possibly, and we did not test it. Zech et al. showed that pneumonia CNNs can pick up hospital- or scanner-specific cues. A saliency or Grad-CAM check on the failure images would be a quick next step. We show the most confident mistakes in the failure-analysis figure but do not claim to know what the model looks at.
