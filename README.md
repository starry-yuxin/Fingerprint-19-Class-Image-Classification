# Fingerprint-19-Class-Image-Classification

This repository implements a 19-class fingerprint image classification pipeline, featuring:
- **K-Fold cross-validation training** (custom stratified k-fold, no sklearn dependency)
- **EMA (Exponential Moving Average)** for more stable inference
- **Mixup / CutMix** augmentation + **AutoAugment (ImageNet policy)** + RandomResizedCrop, etc.
- **TTA (TenCrop / 10-crop)** at inference time, averaging logits over 10 crops
- **K-Fold ensemble**: averaging logits across folds
- **Multi-model fusion (logit-level weighted fusion)** with optional **bias tuning** and optional **grid search** over fusion weights

Scripts:
- Training (torchvision backbones): `train_kfold_imageonly_ema.py`
- Training (timm backbones + AMP): `train_kfold_imageonly_ema_timm.py`
- Inference (KFold ensemble + TTA10): `infer_kfold_ensemble_tta.py`
- Inference (timm inference): `infer_kfold_ensemble_tta_timm.py`
- 3-model fusion (weight sweep + bias): `fuse_3models_sweep_and_bias.py`

---

## 1. Requirements

Recommended: Python >= 3.9, PyTorch >= 2.0.

### Installation
```bash
pip install -U torch torchvision
pip install -U numpy pandas pillow tqdm
pip install -U timm   # required for timm backbones (e.g., SwinV2 / ViT / DeiT)
