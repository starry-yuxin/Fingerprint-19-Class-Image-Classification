import os
import time
import math
import argparse
import random
from typing import Optional, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


# -----------------------
# Utils
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# -----------------------
# Mixup / CutMix
# -----------------------
def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mixup_cutmix(x, y, mixup_alpha, cutmix_alpha):
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return x, y, y, 1.0

    use_cutmix = False
    if mixup_alpha > 0 and cutmix_alpha > 0:
        use_cutmix = (np.random.rand() < 0.5)
    elif cutmix_alpha > 0:
        use_cutmix = True

    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        index = torch.randperm(x.size(0), device=x.device)
        y_a, y_b = y, y[index]
        _, _, H, W = x.size()
        x1, y1, x2, y2 = rand_bbox(W, H, lam)
        x_aug = x.clone()
        x_aug[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
        return x_aug, y_a, y_b, lam

    lam = np.random.beta(mixup_alpha, mixup_alpha)
    index = torch.randperm(x.size(0), device=x.device)
    x_aug = lam * x + (1 - lam) * x[index]
    return x_aug, y, y[index], lam


def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# -----------------------
# Loss
# -----------------------
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, logits, target):
        n = logits.size(1)
        logp = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true = torch.zeros_like(logp)
            true.fill_(self.smoothing / max(n - 1, 1))
            true.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return (-true * logp).sum(dim=1).mean()


# -----------------------
# EMA (SAFE VERSION)
# -----------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.ema_model = self._clone(model)

    def _clone(self, model):
        import copy
        m = copy.deepcopy(model)
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
        return m

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        esd = self.ema_model.state_dict()

        for k in esd.keys():
            if k not in msd:
                continue
            v = msd[k].detach()
            if torch.is_floating_point(esd[k]) and torch.is_floating_point(v):
                esd[k].mul_(d).add_(v, alpha=(1.0 - d))
            else:
                esd[k].copy_(v)

        self.ema_model.load_state_dict(esd, strict=True)

    def state_dict(self):
        return self.ema_model.state_dict()


# -----------------------
# Dataset subset
# -----------------------
class SubsetImageFolder(Dataset):
    def __init__(self, base, indices, transform):
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        path, y = self.base.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), y


# -----------------------
# Stratified K-Fold
# -----------------------
def stratified_kfold_indices(labels, k, seed):
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(labels):
        idxs = np.where(labels == c)[0]
        rng.shuffle(idxs)
        splits = np.array_split(idxs, k)
        for i in range(k):
            folds[i].extend(splits[i].tolist())
    for i in range(k):
        rng.shuffle(folds[i])
    return folds


# -----------------------
# Model builder
# -----------------------
def build_model(backbone, num_classes):
    if backbone == "r50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if backbone == "effv2s":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    raise ValueError("backbone must be r50 or effv2s")


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    c, t = 0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        p = model(x).argmax(1)
        c += (p == y).sum().item()
        t += y.numel()
    return c / max(t, 1)


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--backbone", required=True, choices=["r50", "effv2s"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--img_size", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)

    ap.add_argument("--mixup_alpha", type=float, default=0.1)
    ap.add_argument("--cutmix_alpha", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--use_val_as_train", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Backbone:", args.backbone, "k=", args.k)

    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")

    base_train = datasets.ImageFolder(train_dir)
    if args.use_val_as_train and os.path.isdir(val_dir):
        base_val = datasets.ImageFolder(val_dir)
        base_train.samples += base_val.samples
        base_train.imgs = base_train.samples
        print("Merged train+val:", len(base_train))

    num_classes = len(base_train.classes)
    labels = np.array([y for _, y in base_train.samples])
    folds = stratified_kfold_indices(labels, args.k, args.seed)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    for fi in range(args.k):
        val_idx = folds[fi]
        train_idx = list(set(range(len(base_train))) - set(val_idx))

        train_ds = SubsetImageFolder(base_train, train_idx, train_tf)
        val_ds = SubsetImageFolder(base_train, val_idx, eval_tf)

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=args.workers, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                                num_workers=args.workers)

        model = build_model(args.backbone, num_classes).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        crit = LabelSmoothingCE(args.label_smoothing)

        ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
        best = -1.0

        print(f"\n===== Fold {fi+1}/{args.k} =====")
        for ep in range(1, args.epochs + 1):
            model.train()
            pbar = tqdm(train_loader, ncols=100, desc=f"Fold{fi} Ep{ep}")
            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)
                x_aug, y_a, y_b, lam = mixup_cutmix(
                    x, y, args.mixup_alpha, args.cutmix_alpha
                )
                opt.zero_grad(set_to_none=True)
                loss = mix_criterion(crit, model(x_aug), y_a, y_b, lam)
                loss.backward()
                opt.step()
                if ema:
                    ema.update(model)
                pbar.set_postfix(loss=float(loss.item()))
            sch.step()

            acc_raw = eval_acc(model, val_loader, device)
            acc = acc_raw
            if ema:
                acc_ema = eval_acc(ema.ema_model.to(device), val_loader, device)
                acc = acc_ema
                print(f"Fold{fi} Ep{ep}: val_raw={acc_raw:.4f} val_ema={acc_ema:.4f}")
            else:
                print(f"Fold{fi} Ep{ep}: val_raw={acc_raw:.4f}")

            if acc > best:
                best = acc
                torch.save({
                    "state_dict": model.state_dict(),
                    "model_ema": ema.state_dict() if ema else None,
                    "class_to_idx": base_train.class_to_idx,
                    "img_size": args.img_size,
                    "backbone": args.backbone,
                    "fold": fi,
                }, os.path.join(args.out_dir, f"{args.backbone}_fold{fi}_best.pth"))
                print("  saved best")

    print("=== KFold Training Done ===")


if __name__ == "__main__":
    main()
