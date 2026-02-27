import os
import re
import csv
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
from tqdm import tqdm


def build_model(backbone: str, num_classes: int) -> nn.Module:
    b = backbone.lower()
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
    raise ValueError("backbone must be one of: r50, effv2s, convnext_tiny")


class TTATenCropFromSamples(Dataset):
    def __init__(self, samples, img_size: int = 320):
        self.samples = samples
        resize_size = int(round(img_size * 1.14))
        self.pre_resize = transforms.Resize(resize_size)
        self.ten_crop = transforms.TenCrop(img_size)
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.pre_resize(img)
        crops = self.ten_crop(img)
        crops = torch.stack([self.to_tensor_norm(c) for c in crops], dim=0)  # (10,3,H,W)
        return crops, y, path


@torch.no_grad()
def infer_tta10_logits(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    amp: bool = False
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_paths: List[str] = []
    all_y: List[torch.Tensor] = []

    use_amp = (device == "cuda") and amp

    for crops, y, paths in tqdm(loader, desc="Infer TTA10", ncols=100):
        bsz = crops.size(0)  # B
        crops = crops.view(-1, crops.size(2), crops.size(3), crops.size(4)).to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(crops)
        else:
            logits = model(crops)

        logits = logits.view(bsz, 10, -1).mean(1)  # (B,C)
        all_logits.append(logits.detach().cpu())
        all_paths.extend(list(paths))
        all_y.append(y.detach().cpu())

    logits_all = torch.cat(all_logits, dim=0)
    y_all = torch.cat(all_y, dim=0)
    return logits_all, all_paths, y_all


def _extract_fold_id(path: str) -> int:
    m = re.search(r"fold(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 9999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--backbone", type=str, required=True, choices=["r50", "effv2s", "convnext_tiny"])
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save_probs", action="store_true",
                    help="export prob_0..prob_{C-1} columns (softmax probs)")
    ap.add_argument("--amp", action="store_true",
                    help="use mixed precision for inference (CUDA only)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    test_dir = os.path.join(args.data_root, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test dir not found: {test_dir}")

    test_ds_plain = datasets.ImageFolder(test_dir, transform=None)
    num_classes = len(test_ds_plain.classes)
    print("Classes:", num_classes, "test size:", len(test_ds_plain))

    if not os.path.isdir(args.ckpt_dir):
        raise FileNotFoundError(f"ckpt_dir not found: {args.ckpt_dir}")

    ckpts = []
    for fn in os.listdir(args.ckpt_dir):
        if not fn.endswith(".pth"):
            continue
        low = fn.lower()
        if "fold" not in low or "best" not in low:
            continue
        if args.backbone.lower() not in low:
            continue
        ckpts.append(os.path.join(args.ckpt_dir, fn))

    if len(ckpts) == 0:
        raise FileNotFoundError(f"No fold best checkpoints found in {args.ckpt_dir}")

    ckpts = sorted(ckpts, key=_extract_fold_id)

    print("Found ckpts:")
    for p in ckpts:
        print(" ", p)

    first = torch.load(ckpts[0], map_location="cpu")
    img_size = int(first.get("img_size", 320))
    class_to_idx = first.get("class_to_idx", test_ds_plain.class_to_idx)

    tta_ds = TTATenCropFromSamples(test_ds_plain.samples, img_size=img_size)
    tta_loader = DataLoader(
        tta_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    sum_logits: Optional[torch.Tensor] = None
    all_paths: Optional[List[str]] = None
    all_y: Optional[torch.Tensor] = None

    for p in ckpts:
        ckpt = torch.load(p, map_location="cpu")
        model = build_model(args.backbone, num_classes).to(device)

        if args.use_ema and ckpt.get("model_ema", None) is not None:
            model.load_state_dict(ckpt["model_ema"], strict=True)
            print("Load EMA:", os.path.basename(p))
        else:
            model.load_state_dict(ckpt["state_dict"], strict=True)
            print("Load RAW:", os.path.basename(p))

        logits, paths, y = infer_tta10_logits(model, tta_loader, device, amp=args.amp)

        if sum_logits is None:
            sum_logits = logits
            all_paths = paths
            all_y = y
        else:
            sum_logits += logits

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    assert sum_logits is not None and all_paths is not None and all_y is not None

    avg_logits = sum_logits / float(len(ckpts))
    probs = torch.softmax(avg_logits, dim=1)
    pred = probs.argmax(dim=1)

    acc = (pred == all_y).float().mean().item()
    print(f"KFold Ensemble test_acc(TTA10) = {acc:.4f}")

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    header = ["image_path", "true_class_id", "pred_class_name", "pred_class_id"]
    if args.save_probs:
        header += [f"prob_{i}" for i in range(num_classes)]

    probs_np = probs.cpu().numpy()

    rows = [header]
    for i, (pth, pid) in enumerate(zip(all_paths, pred.tolist())):
        rel = os.path.relpath(pth, start=args.data_root).replace("\\", "/")
        true_id = int(all_y[i].item())
        row = [rel, true_id, idx_to_class[int(pid)], int(pid)]
        if args.save_probs:
            row += [float(probs_np[i, j]) for j in range(num_classes)]
        rows.append(row)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print("Saved:", args.out_csv)


if __name__ == "__main__":
    main()
