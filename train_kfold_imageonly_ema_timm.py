import os
import csv
import math
import time
import argparse
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models


# -----------------------
# Utils
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_float_tensor(x: torch.Tensor) -> bool:
    return torch.is_floating_point(x)


# -----------------------
# Mixup / Cutmix
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


def mixup_cutmix(images, targets, alpha_mixup=0.0, alpha_cutmix=0.0, num_classes=19):
    """
    Returns mixed_images, targets_a(onehot), targets_b(onehot), lam
    If no mix applied: lam=1 and targets_b ignored.
    """
    if alpha_mixup <= 0 and alpha_cutmix <= 0:
        onehot = F.one_hot(targets, num_classes=num_classes).float()
        return images, onehot, onehot, 1.0

    use_cutmix = (alpha_cutmix > 0) and (np.random.rand() < 0.5)
    if use_cutmix:
        lam = np.random.beta(alpha_cutmix, alpha_cutmix)
    else:
        lam = np.random.beta(alpha_mixup, alpha_mixup) if alpha_mixup > 0 else 1.0

    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)
    targets_a = targets
    targets_b = targets[perm]

    if use_cutmix:
        _, _, H, W = images.shape
        x1, y1, x2, y2 = rand_bbox(W, H, lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(W * H))
    else:
        images = images * lam + images[perm] * (1.0 - lam)

    one_a = F.one_hot(targets_a, num_classes=num_classes).float()
    one_b = F.one_hot(targets_b, num_classes=num_classes).float()
    return images, one_a, one_b, float(lam)


# -----------------------
# Label smoothing CE (supports soft targets)
# -----------------------
class SoftTargetCrossEntropy(nn.Module):
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.ls = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets_onehot: torch.Tensor) -> torch.Tensor:
        # logits: (B,C) targets_onehot: (B,C)
        if self.ls > 0:
            C = targets_onehot.size(1)
            targets_onehot = targets_onehot * (1.0 - self.ls) + self.ls / C
        logp = F.log_softmax(logits, dim=1)
        loss = -(targets_onehot * logp).sum(dim=1).mean()
        return loss


# -----------------------
# EMA (safe)
# -----------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.ema_model = self._clone_model(model)

    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        import copy
        m = copy.deepcopy(model)
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        esd = self.ema_model.state_dict()

        for k in esd.keys():
            if k not in msd:
                continue
            v = msd[k].detach()
            if is_float_tensor(esd[k]) and is_float_tensor(v):
                esd[k].mul_(d).add_(v, alpha=(1.0 - d))
            else:
                esd[k].copy_(v)

        self.ema_model.load_state_dict(esd, strict=True)


# -----------------------
# Dataset (list of samples)
# -----------------------
class ImageListDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, y


def collect_samples(data_root: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]], Dict[str, int]]:
    """
    Expect folders:
      data_root/train/class_x/*.png
      data_root/val/class_x/*.png
      data_root/test/class_x/*.png   (has labels in folder name)
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    # Use ImageFolder only to get class_to_idx stable mapping
    from torchvision.datasets import ImageFolder
    train_plain = ImageFolder(train_dir, transform=None)
    class_to_idx = train_plain.class_to_idx

    val_plain = ImageFolder(val_dir, transform=None)
    test_plain = ImageFolder(test_dir, transform=None)

    # Force all to use train's mapping
    def remap(samples, folder_class_to_idx):
        inv = {v: k for k, v in folder_class_to_idx.items()}
        out = []
        for p, y in samples:
            cname = inv[y]
            out.append((p, class_to_idx[cname]))
        return out

    train_samples = remap(train_plain.samples, train_plain.class_to_idx)
    val_samples = remap(val_plain.samples, val_plain.class_to_idx)
    test_samples = remap(test_plain.samples, test_plain.class_to_idx)

    return train_samples, val_samples, test_samples, class_to_idx


def stratified_kfold_indices(y: np.ndarray, k: int, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Simple stratified split without sklearn.
    Returns list of (train_idx, val_idx) for k folds.
    """
    rng = np.random.RandomState(seed)
    cls = np.unique(y)
    per_class = {c: np.where(y == c)[0].tolist() for c in cls}
    for c in cls:
        rng.shuffle(per_class[c])

    folds = [[] for _ in range(k)]
    for c in cls:
        idxs = per_class[c]
        for i, idx in enumerate(idxs):
            folds[i % k].append(idx)

    out = []
    all_idx = set(range(len(y)))
    for f in range(k):
        val_idx = np.array(sorted(folds[f]), dtype=np.int64)
        train_idx = np.array(sorted(list(all_idx - set(folds[f]))), dtype=np.int64)
        out.append((train_idx, val_idx))
    return out


# -----------------------
# Model builder
# -----------------------
def build_model(backbone: str, num_classes: int) -> nn.Module:
    """
    Supports:
      - torchvision: r50, effv2s, convnext_tiny
      - timm: swinv2t, deit_small, vit_small
    """
    b = backbone.lower()

    # torchvision
    if b in ["r50", "resnet50"]:
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if b in ["effv2s", "efficientnet_v2_s"]:
        m = models.efficientnet_v2_s(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    if b in ["convnext_tiny", "cnxt_tiny"]:
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m

    # timm
    try:
        import timm
    except Exception as e:
        raise RuntimeError("timm not installed. Run: pip install timm") from e

    timm_map = {
        "swinv2t": "swinv2_tiny_window16_256.ms_in1k",
        "deit_small": "deit_small_patch16_224.fb_in1k",
        "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    }
    if b in timm_map:
        name = timm_map[b]
        m = timm.create_model(name, pretrained=True, num_classes=num_classes)
        return m

    raise ValueError(f"Unsupported backbone: {backbone}")


# -----------------------
# Train / Eval
# -----------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_one_fold(
    fold_id: int,
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
    class_to_idx: Dict[str, int],
    args
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = len(class_to_idx)

    # transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds_train = ImageListDataset(train_samples, transform=train_tf)
    ds_val = ImageListDataset(val_samples, transform=val_tf)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=(device == "cuda"),
                          drop_last=True, persistent_workers=(args.workers > 0))
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=(device == "cuda"),
                        persistent_workers=(args.workers > 0))

    model = build_model(args.backbone, num_classes).to(device)
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    # cosine
    total_steps = args.epochs * len(dl_train)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))

    criterion = SoftTargetCrossEntropy(label_smoothing=args.label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best = -1.0
    best_path = os.path.join(args.out_dir, f"{args.backbone}_fold{fold_id}_best.pth")

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0

        for x, y in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x, ya, yb, lam = mixup_cutmix(
                x, y,
                alpha_mixup=args.mixup_alpha,
                alpha_cutmix=args.cutmix_alpha,
                num_classes=num_classes
            )

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x)
                targets = ya * lam + yb * (1.0 - lam)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            sched.step()
            running += loss.item()

            if ema is not None:
                ema.update(model)

        # eval
        val_raw = evaluate(model, dl_val, device)
        val_ema = evaluate(ema.ema_model, dl_val, device) if ema is not None else -1.0

        score = val_ema if (ema is not None) else val_raw
        if score > best:
            best = score
            ckpt = {
                "state_dict": model.state_dict(),
                "model_ema": (ema.ema_model.state_dict() if ema is not None else None),
                "backbone": args.backbone,
                "img_size": args.img_size,
                "class_to_idx": class_to_idx,
                "best_val": best,
                "fold": fold_id,
            }
            torch.save(ckpt, best_path)

        dt = time.time() - t0
        print(f"Fold{fold_id} Ep{ep}/{args.epochs} "
              f"loss={running/len(dl_train):.4f} "
              f"val_raw={val_raw:.4f} val_ema={val_ema:.4f} "
              f"best={best:.4f} time={dt:.1f}s")

    print("Saved best:", best_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--backbone", type=str, required=True,
                    choices=["r50", "effv2s", "convnext_tiny", "swinv2t", "deit_small", "vit_small"])
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--mixup_alpha", type=float, default=0.1)
    ap.add_argument("--cutmix_alpha", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--use_val_as_train", action="store_true",
                    help="merge train+val, then do kfold on merged set")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_samples, val_samples, test_samples, class_to_idx = collect_samples(args.data_root)
    if args.use_val_as_train:
        all_samples = train_samples + val_samples
        print("Merged train+val for kfold. total=", len(all_samples))
    else:
        all_samples = train_samples
        print("Use train only for kfold. total=", len(all_samples))

    y = np.array([yy for _, yy in all_samples], dtype=np.int64)
    folds = stratified_kfold_indices(y, k=args.k, seed=args.seed)

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Backbone: {args.backbone} k={args.k} img_size={args.img_size}")
    print(f"mixup_alpha={args.mixup_alpha} cutmix_alpha={args.cutmix_alpha} "
          f"ls={args.label_smoothing} ema_decay={args.ema_decay}")

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        tr = [all_samples[i] for i in tr_idx.tolist()]
        va = [all_samples[i] for i in va_idx.tolist()]
        print(f"\n===== Fold {fold_id+1}/{args.k} | train={len(tr)} val={len(va)} =====")
        train_one_fold(fold_id, tr, va, class_to_idx, args)


if __name__ == "__main__":
    main()
