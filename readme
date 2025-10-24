#  Multimodal Deepfake Detection using FakeAVCeleb

This repository contains the full implementation of a **multimodal deepfake detection system** that integrates both **audio** and **visual** analysis for reliable classification of real and synthetic media.  
The project was developed as part of a research study at the University of Pretoria for **COS700: Advanced Topics in Computer Science**.

---

##  Overview

Most existing deepfake detectors analyse only the **visual** stream, ignoring manipulated audio.  
This project introduces a **fusion-based approach** that combines four specialised models — two visual and two audio — and integrates their learned features using a mid-level fusion network.

The result is a **balanced and robust detection system** capable of identifying deepfakes even when one modality appears authentic.

---

##  Model Architecture

### 1. **Visual Models**

####  ResNet18 + Error Level Analysis (ELA)
- Focus: Pixel-level compression and editing artefacts.  
- Input: 224×224 ELA-transformed frames.  
- Output: 512-dimensional embedding.  
- Pretrained on ImageNet, fine-tuned on FakeAVCeleb frames.  

####  MesoNet
- Focus: Texture-level inconsistencies and unnatural lighting transitions.  
- Input: 256×256 RGB face crops.  
- Output: 128-dimensional embedding.  
- Detects mesoscopic (mid-level) facial irregularities.

---

### 2. **Audio Models**

####  VGG19 on Stacked Spectrograms
- Focus: Spectral and harmonic irregularities in synthetic voices.  
- Input: 3-channel spectrogram image (Mel, MFCC, Delta MFCC).  
- Output: 4096-dimensional embedding.  
- Pretrained on ImageNet, fine-tuned for binary classification.

####  Lightweight Audio CNN
- Focus: Generalisable time–frequency patterns from Mel spectrograms.  
- Input: 128×128 log-Mel spectrograms.  
- Output: 128-dimensional embedding.  
- Lightweight (≈295K parameters) and highly robust to noise.

---

### 3. **Mid-Level Fusion Model**

| Feature Source | Embedding Size |
|-----------------|----------------|
| ResNet18 + ELA | 512 |
| MesoNet | 128 |
| VGG19 | 4096 |
| Audio CNN | 128 |
| **Total Combined** | **4864** |

All embeddings are **L2-normalised** and **concatenated** into a 4864-dimensional feature vector.  
This is passed through a **fully connected MLP** (`4864 → 1024 → 256 → 1`) with dropout and **modality dropout (p=0.15)** to improve generalisation.

The fusion head outputs a **single sigmoid probability** for “real” vs “fake”.

---

##  Training Configuration

- **Optimiser:** AdamW  
- **Loss Function:** Weighted Binary Cross-Entropy (to handle class imbalance)  
- **Learning Rate:**  
  - `1e-4` – `1e-5` for pretrained models (fine-tuning)  
  - `1e-3` for Fusion MLP (trained from scratch)  
- **Batch Size:** 16–32 depending on model  
- **Epochs:** 20–30 with early stopping  
- **Regularisation:** Dropout (0.3) and Modality Dropout (0.15)  
- **Scheduler:** StepLR or CosineAnnealingLR for gradual LR decay  

---

##  Dataset

**FakeAVCeleb** dataset (Oxford University)  
- Includes real and fake videos with both **audio and visual manipulations**.  
- Dataset split is **hash-disjoint** to ensure no subject overlap between train, validation, and test sets.  
- Frame extraction at 1 FPS, audio sampled at 16kHz, 4-second windows.  
- Roughly balanced after augmentation.

---

##  Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|---------|-----------|
| ResNet18 + ELA | 92.9% | High | Lower | Balanced |
| MesoNet | 94.0% | Balanced | Balanced | Good |
| Audio CNN | 96.0% | Balanced | Balanced | Strongest unimodal |
| VGG19 | 90.9% | High | Lower | Good |
| **Fusion (All)** | **97.1%** | **97%** | **97%** | **97%** |

 Fusion improves overall **stability and robustness** rather than raw accuracy alone.

---

##  Evaluation Metrics
- Accuracy  
- Precision, Recall, and F1-Score  
- Confusion Matrix  
- ROC–AUC Curve  

All metrics were calculated at the **video level** by averaging frame/window predictions before thresholding.

---

##  Key Contributions

1. Introduced a **balanced multimodal fusion framework** for deepfake detection.  
2. Demonstrated that combining texture-level, compression-level, and spectral cues improves robustness.  
3. Applied **modality dropout** to make the fusion model resilient to missing or corrupted inputs.  
4. Achieved a balanced 97% precision and recall on FakeAVCeleb.

---

##  Directory Structure
├── ela_ResNet18.ipynb # Visual Model 1 (ELA + ResNet18)
├── mesonet.ipynb # Visual Model 2 (MesoNet)
├── vgg19.ipynb # Audio Model 1 (Stacked Spectrogram)
├── audioCNN.ipynb # Audio Model 2 (Lightweight Mel-CNN)
├── fusion.ipynb # Fusion and Evaluation Pipeline
├── README.md # Project Overview (this file)
└── requirements.txt # Dependencies


---

## Future Work
- Add **temporal attention** for video and audio synchronisation.  
- Evaluate on **cross-dataset generalisation** (e.g., DFDC, FaceForensics++).  
- Integrate **explainability methods** (e.g., Grad-CAM for feature visualisation).  
- Deploy lightweight version for **real-time screening**.

---

## Authors
**Shiza Butt & Ashir Butt**  
*University of Pretoria – COS700 Research Project*  
Supervisor: **Dr Avinash Singh, Dr Thambo Nyathi**

---

## License
This project is released for **academic and research purposes only**.  
Commercial use is prohibited without prior permission.

---


