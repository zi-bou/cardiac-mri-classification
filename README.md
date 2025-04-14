
# 🧠 Cardiac MRI Multi-Task Classifier

This project is a **multi-task deep learning pipeline** using 3D cardiac MRI data.
It performs **simultaneous classification and segmentation** using a shared DenseNet121 3D encoder.

---

## Overview

We train a model to:
- 🏷️ **Classify** cardiac condition (5-class classification)
- 🎯 **Segment** the heart structures (binary segmentation on systole)

### 📊 Input/Output Summary

- **Input**: 2-channel volume ([2, D, H, W]) combining:
  - `frame01.nii` = systole
  - `frameXX.nii` = diastole
- **Outputs**:
  - Classification: vector of 5 logits
  - Segmentation: binary mask of shape [1, D, H, W]

---

## 📂 Data Structure

```
/train/
  ├── p0001/
  │    ├── frame01.nii         # systole
  │    ├── frame01_gt.nii      # segmentation mask (only for frame01)
  │    ├── frame12.nii         # diastole
  │    └── gt.txt              # classification label
  ├── p0002/
  ...
```

---

## 🧰 Dependencies

Install the required libraries (tested in Google Colab):

```bash
!pip uninstall -y numpy numba tensorflow thinc
!pip install numpy==1.23.5
!pip install monai nibabel matplotlib scikit-learn torch torchvision torchaudio -q
```

---

## 🧱 Architecture

```mermaid
graph TD
    A[Load systolic & diastolic MRI] --> B[Preprocess (pad/crop, normalize)]
    B --> C[Stack into 2 channels: [2, D, H, W]]
    C --> D[Shared 3D DenseNet Encoder]
    D --> E1[Classification Head → 5 classes]
    D --> E2[Segmentation Decoder → binary mask]
```

---

## 🧪 Loss Function

- 🔀 **Multi-task loss** = `classification_loss + λ * segmentation_loss`
- `classification_loss` = FocalLoss (handles imbalance)
- `segmentation_loss` = DiceLoss (handles class overlap)

---

## 📈 Performance Tracking

We monitor:
- Accuracy curves
- Loss curves
- Confusion matrix
- Class distribution

Early stopping is triggered after 5 epochs without validation improvement.

---

## 💾 Best Model

After training, the model is saved as:

```
/content/best_model_densenet121_v6.pth
```

---

## 🧠 Author

Made with ❤️ by [Zineb Martinez] - civil engineer turned AI practitioner.
