# 1. Introduction

Pneumonia is an acute infection of the lungs and remains the single largest infectious cause of death in children worldwide. According to the World Health Organization, it killed 740 180 children under the age of five in 2019, accounting for 14% of all deaths in that age group [1]. Most of these deaths are preventable or treatable when the disease is identified early, so fast and reliable diagnosis matters.

The chest X-ray (CXR) is the most common imaging test used to confirm pneumonia, but it is not easy to read. Signs such as patchy opacities or consolidation can be subtle, can overlap with other conditions, and look different in small children than in adults. Reading CXRs well requires trained radiologists, and these specialists are scarce in many of the low-resource settings where the pneumonia burden is highest [3]. This motivates research into automated tools that could support (not replace) clinical readers.

Deep convolutional neural networks (CNNs) have shown strong results on medical images. CheXNet, a 121-layer CNN trained on more than 100 000 chest X-rays, was reported to detect pneumonia at a level comparable to practising radiologists on a held-out test set [3]. Because labelled medical datasets are usually small, most such systems rely on *transfer learning*: a network is first trained on a large natural-image dataset (ImageNet) and then reused for the medical task. Yosinski et al. showed that early CNN layers learn general features (edges, textures) while later layers become specific to the original task, so how much of a network to re-train is an important choice [4]. Raghu et al. further found that for medical images, which differ strongly from natural photos, the benefit of ImageNet features is smaller than often assumed and is concentrated in the lowest layers [5].

Our base paper is Kermany et al. (Cell, 2018) [2]. The authors built a transfer-learning system for retinal OCT scans and then showed it also generalises to classifying pediatric chest X-rays as normal or pneumonia. Their method uses an ImageNet-pretrained InceptionV3 network that is kept completely frozen as a feature extractor; only a new final linear classification layer is trained. They released the dataset (about 5 200 training and 624 test images) publicly, which makes the work well suited for reproduction.

**Our contribution.** In this project we (i) reproduce the frozen-InceptionV3 baseline on the same public dataset and report accuracy, sensitivity, specificity and ROC-AUC, and (ii) test one targeted change: fine-tuning only the last two Inception blocks together with the classifier, so that the highest-level features can adapt to grayscale X-rays. This is a student research-reproduction study; it makes no clinical claims.

## References

[1] World Health Organization, "Pneumonia in children," Fact sheet, Nov. 11, 2022. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/pneumonia

[2] D. S. Kermany *et al.*, "Identifying medical diagnoses and treatable diseases by image-based deep learning," *Cell*, vol. 172, no. 5, pp. 1122–1131, 2018, doi: 10.1016/j.cell.2018.02.010.

[3] P. Rajpurkar *et al.*, "CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning," arXiv:1711.05225, 2017.

[4] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are features in deep neural networks?," in *Advances in Neural Information Processing Systems 27 (NIPS)*, 2014, pp. 3320–3328.

[5] M. Raghu, C. Zhang, J. Kleinberg, and S. Bengio, "Transfusion: Understanding transfer learning for medical imaging," in *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019.
