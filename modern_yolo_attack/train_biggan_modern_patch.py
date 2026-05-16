"""
Train Adversarial Patch chống YOLO đời mới (v8/v9/v10) dùng BigGAN generator.

Tính năng:
  - Generator mode: biggan (BigGAN latent) hoặc raw (pixel trực tiếp)
  - Ensemble attack: tấn công nhiều YOLO cùng lúc để tăng transferability
  - Loss khả vi từ raw head outputs của Ultralytics (v8/v9/v10)
  - Objective: -L_yolo + λ_tv * L_TV + λ_nps * L_NPS
  - LR Scheduler: CosineAnnealingLR
  - Tensorboard logging (tùy chọn)
  - Checkpoint resume
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (
    PROJECT_ROOT,
    PROJECT_ROOT / "adversarialYolo",
    PROJECT_ROOT / "GANLatentDiscovery",
    PROJECT_ROOT / "ultralytics",
):
    _candidate_str = str(_candidate)
    if _candidate.exists() and _candidate_str not in sys.path:
        sys.path.insert(0, _candidate_str)

from adversarialYolo.load_data import InriaDataset, PatchApplier, PatchTransformer

from .bridge import OLD_REPO_ROOT, bootstrap_paths
from .common import (
    DATASET_ROOT,
    DEFAULT_LABEL_DIR_NAME,
    DEFAULT_WEIGHTS_DIR,
    MODEL_CHOICES,
    MODERN_ATTACK_ROOT,
    build_attack_detection_criterion,
    load_modern_detector,
    load_multiple_detectors,
)
from .nps import NPSCalculator

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


# ──────────────────────────────────────────────────────────────────────────────
# Loss modules
# ──────────────────────────────────────────────────────────────────────────────

class TotalVariation(torch.nn.Module):
    """TV loss để patch mượt hơn, giảm nhiễu tần số cao."""

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        tvcomp1 = torch.sum(
            torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 1e-6), dim=0
        )
        tvcomp1 = torch.sum(torch.sum(tvcomp1, dim=0), dim=0)
        tvcomp2 = torch.sum(
            torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 1e-6), dim=0
        )
        tvcomp2 = torch.sum(torch.sum(tvcomp2, dim=0), dim=0)
        return (tvcomp1 + tvcomp2) / torch.numel(adv_patch)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class NoOpSummaryWriter:
    """Fallback writer khi môi trường chưa có tensorboard."""

    def __init__(self, *args, **kwargs) -> None:
        return None

    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def add_image(self, *args, **kwargs) -> None:
        return None

    def add_figure(self, *args, **kwargs) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def make_summary_writer(log_dir: Path):
    if SummaryWriter is None:
        print("[tensorboard] tensorboard chưa được cài. Tắt logging TensorBoard cho run này.")
        return NoOpSummaryWriter()
    print(f"[tensorboard] tensorboard --logdir={log_dir}")
    return SummaryWriter(log_dir=str(log_dir))


def ensure_tensorboardx_compat() -> None:
    """
    Cấp shim `tensorboardX.SummaryWriter` khi môi trường không có package này.

    GANLatentDiscovery import `tensorboardX` ở module scope dù luồng hiện tại chỉ
    cần load pretrained BigGAN. Shim này giúp bỏ dependency đó một cách an toàn.
    """
    try:
        import tensorboardX  # noqa: F401
        return
    except ImportError:
        pass

    fallback_writer = NoOpSummaryWriter
    if SummaryWriter is not None:
        fallback_writer = SummaryWriter

    tensorboardx_module = types.ModuleType("tensorboardX")
    tensorboardx_module.SummaryWriter = fallback_writer
    sys.modules["tensorboardX"] = tensorboardx_module


def ensure_numpy_compat() -> None:
    """Cấp các alias cũ mà GANLatentDiscovery còn dùng."""
    if not hasattr(np, "product"):
        np.product = np.prod


def load_history_rows(history_jsonl_path: Path) -> list[dict]:
    if not history_jsonl_path.exists():
        return []

    rows = []
    with open(history_jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_history_files(
    history_csv_path: Path,
    history_jsonl_path: Path,
    history_rows: list[dict],
) -> None:
    if not history_rows:
        return

    base_fieldnames = [
        "epoch",
        "mean_attack_obj",
        "mean_detector_train_loss",
        "mean_loss_tv",
        "mean_loss_nps",
        "mean_loss_total",
        "lr",
        "steps",
    ]
    extra_fieldnames = sorted(
        {
            key
            for row in history_rows
            for key in row.keys()
            if key not in base_fieldnames
        }
    )
    fieldnames = base_fieldnames + extra_fieldnames

    with open(history_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    with open(history_jsonl_path, "w") as f:
        for row in history_rows:
            f.write(json.dumps(row) + "\n")


def render_history_plot(history_rows: list[dict], history_plot_path: Path) -> bool:
    if plt is None or not history_rows:
        return False

    epochs = [row["epoch"] for row in history_rows]
    total_loss = [row["mean_loss_total"] for row in history_rows]
    attack_obj = [row["mean_attack_obj"] for row in history_rows]
    detector_loss = [row["mean_detector_train_loss"] for row in history_rows]
    tv_loss = [row["mean_loss_tv"] for row in history_rows]
    nps_loss = [row["mean_loss_nps"] for row in history_rows]
    lrs = [row["lr"] for row in history_rows]
    eval_ap50 = [row.get("eval_mean_ap50") for row in history_rows]
    eval_ap50_95 = [row.get("eval_mean_ap50_95") for row in history_rows]
    has_eval_ap = any(value is not None for value in eval_ap50 + eval_ap50_95)

    if has_eval_ap:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(epochs, total_loss, label="total")
    axes[0].plot(epochs, detector_loss, label="detector_train")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, attack_obj, color="tab:red")
    axes[1].set_title("Attack Objective")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, tv_loss, label="tv")
    axes[2].plot(epochs, nps_loss, label="nps")
    axes[2].set_title("Regularization")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    lr_axis_index = 3
    if has_eval_ap:
        ap_epochs = []
        ap50_vals = []
        ap50_95_vals = []
        for row in history_rows:
            if row.get("eval_mean_ap50") is not None:
                ap_epochs.append(row["epoch"])
                ap50_vals.append(row["eval_mean_ap50"])
                ap50_95_vals.append(row["eval_mean_ap50_95"])

        if ap_epochs:
            axes[3].plot(ap_epochs, ap50_vals, 'o-', label="AP50", linewidth=2)
            axes[3].plot(ap_epochs, ap50_95_vals, 's-', label="AP50-95", linewidth=2)
            axes[3].set_title("Evaluation AP")
            axes[3].set_xlabel("Epoch")
            axes[3].grid(True, alpha=0.3)
            axes[3].legend()
        lr_axis_index = 4

    axes[lr_axis_index].plot(epochs, lrs, color="tab:green")
    axes[lr_axis_index].set_title("Learning Rate")
    axes[lr_axis_index].set_xlabel("Epoch")
    axes[lr_axis_index].grid(True, alpha=0.3)

    if has_eval_ap and len(axes) > 5:
        axes[5].axis("off")

    fig.tight_layout()
    fig.savefig(history_plot_path, dpi=160)
    plt.close(fig)
    return True


def build_patch_preview(
    args,
    generator,
    latent_shift: torch.Tensor | None,
    raw_patch: torch.Tensor | None,
) -> torch.Tensor:
    with torch.no_grad():
        if args.generator_mode == "biggan":
            latent_clamped = torch.clamp(
                latent_shift, -args.latent_clip, args.latent_clip
            )
            return torch.clamp(
                (generator(latent_clamped.unsqueeze(0)) + 1) * 0.5, 0.0, 1.0
            )
        return torch.clamp(raw_patch, 0.0, 1.0).unsqueeze(0)


def build_checkpoint_data(
    epoch: int,
    args,
    model_names: list[str],
    optimizer: torch.optim.Optimizer,
    latent_shift: torch.Tensor | None,
    raw_patch: torch.Tensor | None,
    epoch_summary: dict,
) -> dict:
    checkpoint_data = {
        "epoch": epoch,
        "generator_mode": args.generator_mode,
        "models": model_names,
        "optimizer": optimizer.state_dict(),
        "epoch_summary": epoch_summary,
    }
    if args.generator_mode == "biggan":
        checkpoint_data["latent_shift"] = latent_shift.detach().cpu()
        checkpoint_data["class_biggan"] = args.class_biggan
    else:
        checkpoint_data["raw_patch"] = raw_patch.detach().cpu()
    return checkpoint_data


def lab_batch_to_validator_batch(
    img_batch: torch.Tensor,
    lab_batch: torch.Tensor,
) -> dict:
    """
    Chuyển `lab_batch` sang format DetectionValidator cần để tính AP.

    Khác với loss batch, `batch_idx` ở đây phải là tensor 1D.
    """
    device = img_batch.device
    img_h, img_w = img_batch.shape[2:]
    batch_idx_parts = []
    cls_parts = []
    bbox_parts = []

    for image_index in range(lab_batch.size(0)):
        labels_i = lab_batch[image_index]
        valid_mask = labels_i[:, 0] == 0
        valid_labels = labels_i[valid_mask]
        if valid_labels.numel() == 0:
            continue

        count = valid_labels.size(0)
        batch_idx_parts.append(
            torch.full((count,), image_index, device=device, dtype=torch.long)
        )
        cls_parts.append(valid_labels[:, 0:1].to(device=device, dtype=torch.float32))
        bbox_parts.append(
            valid_labels[:, 1:5].to(device=device, dtype=torch.float32).clamp_(0.0, 1.0)
        )

    if batch_idx_parts:
        batch_idx = torch.cat(batch_idx_parts, dim=0)
        cls = torch.cat(cls_parts, dim=0)
        bboxes = torch.cat(bbox_parts, dim=0)
    else:
        batch_idx = torch.empty((0,), device=device, dtype=torch.long)
        cls = torch.empty((0, 1), device=device, dtype=torch.float32)
        bboxes = torch.empty((0, 4), device=device, dtype=torch.float32)

    return {
        "img": img_batch,
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
        "ori_shape": [(img_h, img_w)] * img_batch.size(0),
        "ratio_pad": [((1.0, 1.0), (0.0, 0.0))] * img_batch.size(0),
    }


def build_ap_validator(
    detector,
    image_size: int,
    save_dir: Path,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
):
    """Khởi tạo DetectionValidator để tính AP trên ảnh đã dán patch."""
    bootstrap_paths()
    from ultralytics.models.yolo.detect import DetectionValidator

    validator = DetectionValidator(
        args={
            "imgsz": image_size,
            "conf": conf_threshold,
            "iou": iou_threshold,
            "max_det": max_det,
            "single_cls": False,
            "save_json": False,
            "save_txt": False,
            "save_conf": False,
            "save_hybrid": False,
            "plots": False,
            "verbose": False,
            "split": "val",
            "task": "detect",
            "half": False,
            "augment": False,
            "agnostic_nms": False,
            "classes": None,
        },
        save_dir=save_dir,
    )
    validator.device = next(detector.model.parameters()).device
    validator.data = {"val": ""}
    validator.training = False
    validator.init_metrics(detector.model)
    return validator


def reset_detector_inference_cache(detector) -> None:
    """
    Xóa cache inference tensors trong detect head để tránh xung đột với autograd
    khi train tiếp sau một pha eval.
    """
    head = detector.model.model[-1]
    if hasattr(head, "shape"):
        head.shape = None
    if hasattr(head, "anchors"):
        head.anchors = torch.empty(0, device=head.anchors.device, dtype=head.anchors.dtype)
    if hasattr(head, "strides"):
        head.strides = torch.empty(0, device=head.strides.device, dtype=head.strides.dtype)


def evaluate_detector_ap(
    detector,
    model_name: str,
    eval_dataloader: DataLoader,
    adv_patch_img: torch.Tensor,
    patch_transformer,
    patch_applier,
    image_size: int,
    patch_scale: float,
    device: torch.device,
    save_dir: Path,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    max_batches: int = 0,
) -> dict[str, float | int]:
    """Tính AP của một detector trên tập eval sau khi dán patch."""
    validator = build_ap_validator(
        detector=detector,
        image_size=image_size,
        save_dir=save_dir / model_name,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_det=max_det,
    )
    detector.model.eval()
    try:
        with torch.no_grad():
            for batch_idx, (img_batch, lab_batch) in enumerate(eval_dataloader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break

                img_batch = img_batch.to(device)
                lab_batch = lab_batch.to(device)
                adv_batch_t, _, _ = patch_transformer(
                    adv_patch=adv_patch_img,
                    lab_batch=lab_batch,
                    img_size=image_size,
                    patch_mask=[],
                    by_rectangle=True,
                    do_rotate=False,
                    rand_loc=False,
                    with_black_trans=False,
                    scale_rate=patch_scale,
                    with_crease=False,
                    with_projection=False,
                    with_rectOccluding=False,
                    enable_blurred=False,
                )
                patched_imgs = patch_applier(img_batch, adv_batch_t)
                preds = validator.postprocess(detector.model(patched_imgs))
                validator_batch = lab_batch_to_validator_batch(patched_imgs, lab_batch)
                validator.update_metrics(preds, validator_batch)

        stats = validator.get_stats()
        validator.finalize_metrics()
        instances = int(validator.nt_per_class.sum()) if validator.nt_per_class is not None else 0
        return {
            "precision": float(stats.get("metrics/precision(B)", 0.0)),
            "recall": float(stats.get("metrics/recall(B)", 0.0)),
            "ap50": float(stats.get("metrics/mAP50(B)", 0.0)),
            "ap50_95": float(stats.get("metrics/mAP50-95(B)", 0.0)),
            "fitness": float(stats.get("fitness", 0.0)),
            "images": int(validator.seen or 0),
            "instances": instances,
        }
    finally:
        reset_detector_inference_cache(detector)


def evaluate_detectors_ap(
    detector_entries: list[tuple],
    model_names: list[str],
    eval_dataloader: DataLoader,
    adv_patch_img: torch.Tensor,
    patch_transformer,
    patch_applier,
    image_size: int,
    patch_scale: float,
    device: torch.device,
    save_dir: Path,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    max_batches: int = 0,
) -> dict:
    """Tính AP cho từng detector và thêm mean AP cho ensemble/single."""
    results = {}
    ap50_values = []
    ap50_95_values = []

    for model_name, (detector, _, _) in zip(model_names, detector_entries):
        detector_result = evaluate_detector_ap(
            detector=detector,
            model_name=model_name,
            eval_dataloader=eval_dataloader,
            adv_patch_img=adv_patch_img,
            patch_transformer=patch_transformer,
            patch_applier=patch_applier,
            image_size=image_size,
            patch_scale=patch_scale,
            device=device,
            save_dir=save_dir,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            max_batches=max_batches,
        )
        results[model_name] = detector_result
        ap50_values.append(detector_result["ap50"])
        ap50_95_values.append(detector_result["ap50_95"])

    results["mean"] = {
        "ap50": float(sum(ap50_values) / len(ap50_values)) if ap50_values else 0.0,
        "ap50_95": float(sum(ap50_95_values) / len(ap50_95_values)) if ap50_95_values else 0.0,
    }
    return results


def lab_batch_to_ultralytics_batch(
    img_batch: torch.Tensor,
    lab_batch: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Chuyển `lab_batch` từ InriaDataset sang format batch mà Ultralytics loss cần.

    InriaDataset pad các dòng rỗng bằng toàn số 1, nên chỉ giữ các box có cls_id=0.
    """
    device = img_batch.device
    batch_idx_parts = []
    cls_parts = []
    bbox_parts = []

    for image_index in range(lab_batch.size(0)):
        labels_i = lab_batch[image_index]
        valid_mask = labels_i[:, 0] == 0
        valid_labels = labels_i[valid_mask]
        if valid_labels.numel() == 0:
            continue

        count = valid_labels.size(0)
        batch_idx_parts.append(
            torch.full((count, 1), image_index, device=device, dtype=torch.float32)
        )
        cls_parts.append(valid_labels[:, 0:1].to(device=device, dtype=torch.float32))
        bbox_parts.append(
            valid_labels[:, 1:5].to(device=device, dtype=torch.float32).clamp_(0.0, 1.0)
        )

    if batch_idx_parts:
        batch_idx = torch.cat(batch_idx_parts, dim=0)
        cls = torch.cat(cls_parts, dim=0)
        bboxes = torch.cat(bbox_parts, dim=0)
    else:
        batch_idx = torch.empty((0, 1), device=device, dtype=torch.float32)
        cls = torch.empty((0, 1), device=device, dtype=torch.float32)
        bboxes = torch.empty((0, 4), device=device, dtype=torch.float32)

    return {
        "img": img_batch,
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
    }


def extract_preds_for_attack(raw_outputs):
    """
    Lấy phần output còn gradient về input/patch.

    - YOLOv8/v9 eval path: `(decoded, feats)` -> dùng nguyên tuple.
    - YOLOv10 eval path: `(decoded, {'one2many': ..., 'one2one': ...})`
      -> chỉ dùng `one2many` vì `one2one` bị detach khỏi backbone/input.
    """
    if (
        isinstance(raw_outputs, tuple)
        and len(raw_outputs) == 2
        and isinstance(raw_outputs[1], dict)
        and "one2many" in raw_outputs[1]
    ):
        return raw_outputs[1]["one2many"]
    return raw_outputs


def detector_attack_objective(
    detector,
    criterion,
    patched_imgs: torch.Tensor,
    yolo_batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Trả về:
      - attack objective để MINIMIZE (`-detector_train_loss`)
      - detector train loss dương để logging
    """
    raw_outputs = detector.model(patched_imgs)
    preds_for_loss = extract_preds_for_attack(raw_outputs)
    detector_train_loss, _ = criterion(preds_for_loss, yolo_batch)
    detector_train_loss = detector_train_loss / max(patched_imgs.size(0), 1)
    attack_objective = -detector_train_loss
    return attack_objective, detector_train_loss.detach()


def ensemble_attack_objective(
    detector_entries: list[tuple],
    patched_imgs: torch.Tensor,
    yolo_batch: dict[str, torch.Tensor],
    device: torch.device,
    weights: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tổng hợp objective tấn công từ nhiều detector."""
    n = len(detector_entries)
    if weights is None:
        weights = [1.0 / n] * n

    total_attack = torch.tensor(0.0, device=device)
    total_detector_train_loss = torch.tensor(0.0, device=device)
    for (detector, _, criterion), weight in zip(detector_entries, weights):
        attack_i, detector_train_loss_i = detector_attack_objective(
            detector, criterion, patched_imgs, yolo_batch
        )
        total_attack = total_attack + weight * attack_i
        total_detector_train_loss = total_detector_train_loss + weight * detector_train_loss_i
    return total_attack, total_detector_train_loss


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_biggan(
    old_repo_root: Path,
    class_biggan: int,
    device: torch.device,
):
    """Load BigGAN generator từ GANLatentDiscovery."""
    # Thêm old repo vào sys.path để import GANLatentDiscovery
    old_repo_str = str(old_repo_root)
    if old_repo_str not in sys.path:
        sys.path.insert(0, old_repo_str)

    from GANLatentDiscovery.models.gan_load import make_big_gan

    generator_root = old_repo_root / "GANLatentDiscovery" / "models" / "pretrained"
    generator_weights = generator_root / "generators" / "BigGAN" / "G_ema.pth"

    if not generator_weights.exists():
        raise FileNotFoundError(
            f"BigGAN weights không tìm thấy tại:\n  {generator_weights}\n"
            f"Chạy: bash download_biggan.sh"
        )

    previous_cwd = Path.cwd()
    try:
        os.chdir(old_repo_root)
        generator = make_big_gan(str(generator_weights), class_biggan)
    finally:
        os.chdir(previous_cwd)

    generator = generator.to(device).eval()
    # Tắt gradient của generator (chỉ tối ưu latent_shift)
    for param in generator.parameters():
        param.requires_grad_(False)

    return generator


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train adversarial patch (BigGAN/raw) chống YOLO v8/v9/v10.\n"
            "Hỗ trợ ensemble attack (nhiều model cùng lúc)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Generator
    parser.add_argument(
        "--generator-mode", choices=("biggan", "raw"), default="biggan",
        help="biggan: tối ưu latent vector của BigGAN; raw: tối ưu pixel trực tiếp",
    )
    parser.add_argument(
        "--class-biggan", type=int, default=259,
        help="ImageNet class index cho BigGAN (mặc định 259=Pomeranian, 0=tuck giảm nhận diện người)",
    )
    parser.add_argument("--latent-clip", type=float, default=3.0)
    parser.add_argument("--raw-patch-size", type=int, default=128)

    # Model(s) – hỗ trợ một hoặc nhiều model cho ensemble
    parser.add_argument(
        "--model", choices=MODEL_CHOICES, default="yolov8n",
        help="Model đơn lẻ (dùng khi không có --ensemble-models)",
    )
    parser.add_argument(
        "--ensemble-models", nargs="+", choices=MODEL_CHOICES, default=None,
        metavar="MODEL",
        help=(
            "Danh sách models cho ensemble attack, VD: yolov8n yolov9t yolov10n. "
            "Khi được đặt, --model bị bỏ qua."
        ),
    )

    # Dataset
    parser.add_argument("--label-dir-name", default=DEFAULT_LABEL_DIR_NAME)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--limit", type=int, default=0, help="0 = dùng toàn bộ dataset")
    parser.add_argument("--batch-size", type=int, default=4)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0, help="0 = không giới hạn")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--lr-scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--seed", type=int, default=15089)
    parser.add_argument("--train-conf", type=float, default=0.001)
    parser.add_argument(
        "--eval-metric", choices=("ap", "none"), default="ap",
        help="Metric đánh giá. AP được tính trên ảnh đã dán patch, lower = attack mạnh hơn.",
    )
    parser.add_argument("--eval-every", type=int, default=1,
                        help="Chạy eval metric mỗi N epoch")
    parser.add_argument("--eval-split", choices=("Train", "Test"), default="Test")
    parser.add_argument("--eval-limit", type=int, default=0,
                        help="0 = dùng toàn bộ eval split")
    parser.add_argument("--eval-batch-size", type=int, default=0,
                        help="0 = dùng cùng batch-size train")
    parser.add_argument("--eval-max-batches", type=int, default=0,
                        help="0 = không giới hạn số batch eval")
    parser.add_argument("--eval-conf", type=float, default=0.001,
                        help="Confidence threshold cho AP evaluation")
    parser.add_argument("--eval-iou", type=float, default=0.6,
                        help="IoU threshold cho NMS khi eval AP")
    parser.add_argument("--eval-max-det", type=int, default=300,
                        help="Số detection tối đa giữ lại mỗi ảnh khi eval AP")

    # Loss weights
    parser.add_argument("--weight-loss-det", type=float, default=1.0,
                        help="Trọng số objective tấn công từ detector")
    parser.add_argument("--weight-loss-tv", type=float, default=0.1,
                        help="Trọng số TV loss (patch smooth)")
    parser.add_argument("--weight-loss-nps", type=float, default=0.01,
                        help="Trọng số NPS loss (printability, 0=tắt)")
    parser.add_argument("--patch-scale", type=float, default=0.2,
                        help="Tỉ lệ kích thước patch so với bounding box")

    # Checkpoint resume
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Đường dẫn tới checkpoint .pt để tiếp tục train",
    )

    # Output
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--save-interval", type=int, default=10,
        help="Lưu checkpoint & generated image mỗi N epoch",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Script này yêu cầu CUDA.")

    set_seed(args.seed)
    device = torch.device("cuda")

    # ── Xác định danh sách models ──
    model_names = args.ensemble_models if args.ensemble_models else [args.model]
    is_ensemble = len(model_names) > 1
    print(f"[mode] {'ENSEMBLE' if is_ensemble else 'SINGLE'}: {model_names}")
    print(f"[generator] {args.generator_mode}")

    # ── Dataset ──
    img_dir = DATASET_ROOT / "Train" / "pos"
    lab_dir = img_dir / args.label_dir_name
    if not lab_dir.exists():
        raise FileNotFoundError(
            f"Label directory không tìm thấy: {lab_dir}\n"
            "Chạy: python -m modern_yolo_attack.prepare_inria_labels"
        )

    # ── Output directories ──
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_tag = "_".join(model_names)
    output_dir = args.output_dir or (
        MODERN_ATTACK_ROOT / "exp" / f"{timestamp}_{model_tag}_{args.generator_mode}"
    )
    generated_dir = output_dir / "generated"
    checkpoints_dir = output_dir / "checkpoints"
    history_csv_path = output_dir / "history.csv"
    history_jsonl_path = output_dir / "history.jsonl"
    history_plot_path = output_dir / "history.png"
    best_patch_path = generated_dir / "patch-best.png"
    last_patch_path = generated_dir / "patch-last.png"
    best_checkpoint_path = checkpoints_dir / "checkpoint-best.pt"
    last_checkpoint_path = checkpoints_dir / "checkpoint-last.pt"
    eval_dir = output_dir / "eval"
    tb_dir = MODERN_ATTACK_ROOT / "runs" / "tensorboard" / output_dir.name
    generated_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Lưu args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # ── Tensorboard ──
    writer = make_summary_writer(tb_dir)

    # ── Load detectors ──
    print("\n[detectors] Đang tải YOLO models...")
    if is_ensemble:
        detector_pairs = load_multiple_detectors(model_names, device, args.weights_dir)
    else:
        det, wpath = load_modern_detector(model_names[0], device, args.weights_dir)
        detector_pairs = [(det, wpath)]
    detectors = [
        (detector, weights_path, build_attack_detection_criterion(detector))
        for detector, weights_path in detector_pairs
    ]

    # ── Generator / optimizable parameter ──
    generator = None
    latent_shift = None
    raw_patch = None
    start_epoch = 1
    history_rows = load_history_rows(history_jsonl_path)
    best_loss_total = min(
        (row["mean_loss_total"] for row in history_rows),
        default=float("inf"),
    )
    best_checkpoint_metric_name = (
        "eval_mean_ap50_95" if args.eval_metric == "ap" else "mean_loss_total"
    )
    best_checkpoint_metric_value = min(
        (
            row[best_checkpoint_metric_name]
            for row in history_rows
            if row.get(best_checkpoint_metric_name) is not None
        ),
        default=float("inf"),
    )
    best_epoch = None
    if history_rows:
        candidate_rows = [
            row for row in history_rows if row.get(best_checkpoint_metric_name) is not None
        ]
        if candidate_rows:
            best_row = min(candidate_rows, key=lambda row: row[best_checkpoint_metric_name])
            best_epoch = best_row["epoch"]

    if args.generator_mode == "biggan":
        generator = load_biggan(OLD_REPO_ROOT, args.class_biggan, device)
        latent_shift = torch.normal(
            0.0, torch.ones(generator.dim_z, device=device)
        ).requires_grad_(True)
        optimizer = torch.optim.Adam(
            [latent_shift], lr=args.learning_rate, betas=(0.9, 0.999), amsgrad=True
        )
    else:
        raw_patch = torch.rand(
            (3, args.raw_patch_size, args.raw_patch_size), device=device
        ).requires_grad_(True)
        optimizer = torch.optim.Adam(
            [raw_patch], lr=args.learning_rate, betas=(0.9, 0.999), amsgrad=True
        )

    # ── Resume từ checkpoint ──
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Checkpoint không tìm thấy: {args.resume}")
        print(f"[resume] Tải checkpoint từ {args.resume}")
        ckpt = torch.load(str(args.resume), map_location=device)
        if args.generator_mode == "biggan" and "latent_shift" in ckpt:
            latent_shift.data.copy_(ckpt["latent_shift"].to(device))
        elif args.generator_mode == "raw" and "raw_patch" in ckpt:
            raw_patch.data.copy_(ckpt["raw_patch"].to(device))
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[resume] Tiếp tục từ epoch {start_epoch}")

    # ── LR Scheduler ──
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.learning_rate * 0.01,
        )

    # ── Dataset & DataLoader ──
    dataset = InriaDataset(
        str(img_dir), str(lab_dir), max_lab=20, imgsize=args.image_size, shuffle=True
    )
    if args.limit and args.limit > 0:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
        pin_memory=True,
    )
    print(f"\n[dataset] {len(dataset)} ảnh, {len(dataloader)} batches/epoch")

    eval_dataloader = None
    if args.eval_metric == "ap":
        eval_img_dir = DATASET_ROOT / args.eval_split / "pos"
        eval_lab_dir = eval_img_dir / args.label_dir_name
        if not eval_lab_dir.exists():
            raise FileNotFoundError(
                f"Eval label directory không tìm thấy: {eval_lab_dir}\n"
                "Chạy: python -m modern_yolo_attack.prepare_inria_labels"
            )
        eval_dataset = InriaDataset(
            str(eval_img_dir),
            str(eval_lab_dir),
            max_lab=20,
            imgsize=args.image_size,
            shuffle=False,
        )
        if args.eval_limit and args.eval_limit > 0:
            eval_dataset = Subset(eval_dataset, range(min(args.eval_limit, len(eval_dataset))))
        eval_batch_size = args.eval_batch_size or args.batch_size
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        print(
            f"[eval] AP trên split={args.eval_split}: "
            f"{len(eval_dataset)} ảnh, {len(eval_dataloader)} batches"
        )

    # ── Loss modules ──
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
    total_variation = TotalVariation().cuda()
    nps_calculator = NPSCalculator().cuda() if args.weight_loss_nps > 0 else None

    # ──────────────────────────────────────────────────────────────────────────
    # Training Loop
    # ──────────────────────────────────────────────────────────────────────────
    final_epoch_index = start_epoch + args.epochs - 1
    print(f"\n[training] Bắt đầu từ epoch {start_epoch}/{final_epoch_index}")

    for epoch in range(start_epoch, final_epoch_index + 1):
        epoch_attack_obj = 0.0
        epoch_detector_train_loss = 0.0
        epoch_loss_tv = 0.0
        epoch_loss_nps = 0.0
        epoch_loss_total = 0.0
        epoch_steps = 0

        for batch_idx, (img_batch, lab_batch) in enumerate(dataloader):
            if args.max_batches_per_epoch > 0 and batch_idx >= args.max_batches_per_epoch:
                break

            img_batch = img_batch.to(device)
            lab_batch = lab_batch.to(device)
            optimizer.zero_grad()
            yolo_batch = lab_batch_to_ultralytics_batch(img_batch, lab_batch)

            # ── Generate patch ──
            if args.generator_mode == "biggan":
                latent = torch.clamp(latent_shift, -args.latent_clip, args.latent_clip)
                with torch.enable_grad():
                    fake_images = generator(latent.unsqueeze(0))
                fake_images = torch.clamp((fake_images + 1) * 0.5, 0.0, 1.0)
            else:
                fake_images = torch.clamp(raw_patch, 0.0, 1.0).unsqueeze(0)

            adv_patch_img = fake_images[0]  # (3, H, W)

            # ── Apply patch to images ──
            adv_batch_t, _, _ = patch_transformer(
                adv_patch=adv_patch_img,
                lab_batch=lab_batch,
                img_size=args.image_size,
                patch_mask=[],
                by_rectangle=True,
                do_rotate=False,
                rand_loc=False,
                with_black_trans=False,
                scale_rate=args.patch_scale,
                with_crease=False,
                with_projection=False,
                with_rectOccluding=False,
                enable_blurred=False,
            )
            patched_imgs = patch_applier(img_batch, adv_batch_t)

            # ── Detection attack objective ──
            if is_ensemble:
                attack_obj, detector_train_loss = ensemble_attack_objective(
                    detectors, patched_imgs, yolo_batch, device=device
                )
            else:
                attack_obj, detector_train_loss = detector_attack_objective(
                    detectors[0][0], detectors[0][2], patched_imgs, yolo_batch
                )

            # ── Regularization losses ──
            loss_tv = total_variation(adv_patch_img)
            loss_nps = (
                nps_calculator(adv_patch_img) if nps_calculator is not None
                else torch.tensor(0.0, device=device)
            )

            # ── Total loss ──
            loss = (
                args.weight_loss_det * attack_obj
                + args.weight_loss_tv * loss_tv
                + args.weight_loss_nps * loss_nps
            )
            loss.backward()
            optimizer.step()

            epoch_attack_obj += float(attack_obj.detach().cpu())
            epoch_detector_train_loss += float(detector_train_loss.detach().cpu())
            epoch_loss_tv += float(loss_tv.detach().cpu())
            epoch_loss_nps += float(loss_nps.detach().cpu())
            epoch_loss_total += float(loss.detach().cpu())
            epoch_steps += 1

        # ── LR step ──
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Epoch summary ──
        n = max(epoch_steps, 1)
        epoch_summary = {
            "epoch": epoch,
            "mean_attack_obj": epoch_attack_obj / n,
            "mean_detector_train_loss": epoch_detector_train_loss / n,
            "mean_loss_tv": epoch_loss_tv / n,
            "mean_loss_nps": epoch_loss_nps / n,
            "mean_loss_total": epoch_loss_total / n,
            "lr": current_lr,
            "steps": epoch_steps,
        }

        # ── Save patch image & checkpoints ──
        patch_preview = build_patch_preview(args, generator, latent_shift, raw_patch)
        should_eval_ap = (
            args.eval_metric == "ap"
            and eval_dataloader is not None
            and (epoch % args.eval_every == 0 or epoch == final_epoch_index)
        )
        if should_eval_ap:
            eval_results = evaluate_detectors_ap(
                detector_entries=detectors,
                model_names=model_names,
                eval_dataloader=eval_dataloader,
                adv_patch_img=patch_preview[0].to(device),
                patch_transformer=patch_transformer,
                patch_applier=patch_applier,
                image_size=args.image_size,
                patch_scale=args.patch_scale,
                device=device,
                save_dir=eval_dir,
                conf_threshold=args.eval_conf,
                iou_threshold=args.eval_iou,
                max_det=args.eval_max_det,
                max_batches=args.eval_max_batches,
            )
            epoch_summary["eval_mean_ap50"] = eval_results["mean"]["ap50"]
            epoch_summary["eval_mean_ap50_95"] = eval_results["mean"]["ap50_95"]
            for model_name in model_names:
                model_result = eval_results[model_name]
                epoch_summary[f"eval_precision_{model_name}"] = model_result["precision"]
                epoch_summary[f"eval_recall_{model_name}"] = model_result["recall"]
                epoch_summary[f"eval_ap50_{model_name}"] = model_result["ap50"]
                epoch_summary[f"eval_ap50_95_{model_name}"] = model_result["ap50_95"]
                epoch_summary[f"eval_images_{model_name}"] = model_result["images"]
                epoch_summary[f"eval_instances_{model_name}"] = model_result["instances"]

        print(json.dumps(epoch_summary))
        history_rows.append(epoch_summary)
        write_history_files(history_csv_path, history_jsonl_path, history_rows)
        render_history_plot(history_rows, history_plot_path)

        # ── Tensorboard ──
        writer.add_scalar("loss/attack_obj", epoch_attack_obj / n, epoch)
        writer.add_scalar("loss/detector_train", epoch_detector_train_loss / n, epoch)
        writer.add_scalar("loss/tv", epoch_loss_tv / n, epoch)
        writer.add_scalar("loss/nps", epoch_loss_nps / n, epoch)
        writer.add_scalar("loss/total", epoch_loss_total / n, epoch)
        writer.add_scalar("lr", current_lr, epoch)
        if "eval_mean_ap50" in epoch_summary:
            writer.add_scalar("eval/mean_ap50", epoch_summary["eval_mean_ap50"], epoch)
            writer.add_scalar("eval/mean_ap50_95", epoch_summary["eval_mean_ap50_95"], epoch)
            for model_name in model_names:
                writer.add_scalar(
                    f"eval/{model_name}/ap50",
                    epoch_summary[f"eval_ap50_{model_name}"],
                    epoch,
                )
                writer.add_scalar(
                    f"eval/{model_name}/ap50_95",
                    epoch_summary[f"eval_ap50_95_{model_name}"],
                    epoch,
                )

        checkpoint_data = build_checkpoint_data(
            epoch=epoch,
            args=args,
            model_names=model_names,
            optimizer=optimizer,
            latent_shift=latent_shift,
            raw_patch=raw_patch,
            epoch_summary=epoch_summary,
        )
        save_image(patch_preview, last_patch_path)
        torch.save(checkpoint_data, last_checkpoint_path)

        best_loss_total = min(best_loss_total, epoch_summary["mean_loss_total"])
        current_best_metric_value = epoch_summary.get(best_checkpoint_metric_name)
        is_best = (
            current_best_metric_value is not None
            and current_best_metric_value < best_checkpoint_metric_value
        )
        if is_best:
            best_checkpoint_metric_value = current_best_metric_value
            best_epoch = epoch
            save_image(patch_preview, best_patch_path)
            torch.save(checkpoint_data, best_checkpoint_path)

        if epoch % args.save_interval == 0 or epoch == final_epoch_index:
            save_image(patch_preview, generated_dir / f"patch-epoch-{epoch:04d}.png")
            torch.save(
                checkpoint_data, checkpoints_dir / f"checkpoint-{epoch:04d}.pt"
            )

        # Log patch image vào tensorboard
        writer.add_image("patch", patch_preview[0], epoch)

    # ── Final summary ──
    final_epoch = history_rows[-1] if history_rows else None
    best_epoch_metrics = next(
        (row for row in history_rows if row.get("epoch") == best_epoch),
        None,
    )
    final_summary = {
        "models": model_names,
        "ensemble": is_ensemble,
        "generator_mode": args.generator_mode,
        "epochs_trained": args.epochs,
        "output_dir": str(output_dir),
        "generated_dir": str(generated_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "eval_dir": str(eval_dir),
        "history_csv": str(history_csv_path),
        "history_jsonl": str(history_jsonl_path),
        "history_plot": str(history_plot_path),
        "history_plot_written": history_plot_path.exists(),
        "best_patch": str(best_patch_path),
        "last_patch": str(last_patch_path),
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "best_epoch": best_epoch,
        "best_checkpoint_metric_name": best_checkpoint_metric_name,
        "best_checkpoint_metric_mode": "min",
        "best_checkpoint_metric_value": (
            best_checkpoint_metric_value if best_epoch is not None else None
        ),
        "best_mean_loss_total": best_loss_total if history_rows else None,
        "best_epoch_metrics": best_epoch_metrics,
        "last_epoch_metrics": final_epoch,
        "tensorboard_dir": str(tb_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)
    print("\n" + json.dumps(final_summary, indent=2))

    writer.close()
    print("\n[done] Training hoàn tất!")


if __name__ == "__main__":
    main()
