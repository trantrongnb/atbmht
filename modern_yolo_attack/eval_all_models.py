"""
Đánh giá patch trên TẤT CẢ model variants (yolov8n/s/m/l, yolov9t/s/m/c, yolov10n/s/m/l)
Xuất bảng kết quả tổng hợp để so sánh.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

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

from .common import (
    DATASET_ROOT,
    DEFAULT_LABEL_DIR_NAME,
    DEFAULT_WEIGHTS_DIR,
    MODEL_CHOICES,
    MODERN_ATTACK_ROOT,
    load_modern_detector,
    summarize_person_result,
    to_square_patch_tensor,
)


def eval_single_model(
    model_name: str,
    patch_tensor: torch.Tensor,
    patch_transformer: PatchTransformer,
    patch_applier: PatchApplier,
    dataloader: DataLoader,
    device: torch.device,
    args,
) -> dict:
    """Đánh giá một model, trả về dict kết quả."""
    try:
        detector, weights_path = load_modern_detector(model_name, device, args.weights_dir)
    except Exception as e:
        return {"model": model_name, "error": str(e), "skipped": True}

    total_clean_conf = 0.0
    total_patched_conf = 0.0
    total_clean_count = 0
    total_patched_count = 0
    fooled_zero = 0
    n_images = 0

    for clean_imgs, labels in dataloader:
        clean_imgs = clean_imgs.to(device)
        labels = labels.to(device)

        clean_results = detector(clean_imgs, verbose=False, conf=args.conf_thres)

        adv_batch_t, _, _ = patch_transformer(
            adv_patch=patch_tensor[0],
            lab_batch=labels,
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
            enable_empty_patch=False,
            enable_no_random=True,
            enable_blurred=False,
        )
        patched_imgs = patch_applier(clean_imgs, adv_batch_t)
        patched_results = detector(patched_imgs, verbose=False, conf=args.conf_thres)

        for i in range(clean_imgs.size(0)):
            c = summarize_person_result(clean_results[i], args.conf_thres)
            p = summarize_person_result(patched_results[i], args.conf_thres)
            total_clean_conf += c["max_person_conf"]
            total_patched_conf += p["max_person_conf"]
            total_clean_count += c["person_count_above_threshold"]
            total_patched_count += p["person_count_above_threshold"]
            if p["person_count_above_threshold"] == 0:
                fooled_zero += 1
            n_images += 1

    if n_images == 0:
        return {"model": model_name, "error": "No images processed", "skipped": True}

    mean_clean = total_clean_conf / n_images
    mean_patched = total_patched_conf / n_images
    fooled_rate = fooled_zero / n_images * 100.0

    return {
        "model": model_name,
        "n_images": n_images,
        "mean_clean_max_conf": round(mean_clean, 4),
        "mean_patched_max_conf": round(mean_patched, 4),
        "mean_conf_reduction": round(mean_clean - mean_patched, 4),
        "fooled_rate_pct": round(fooled_rate, 2),
        "mean_clean_count": round(total_clean_count / n_images, 2),
        "mean_patched_count": round(total_patched_count / n_images, 2),
        "skipped": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đánh giá patch trên tất cả YOLOv8/v9/v10 variants."
    )
    parser.add_argument("--patch", type=Path, required=True, help="Đường dẫn tới patch image")
    parser.add_argument("--split", choices=("Train", "Test"), default="Test")
    parser.add_argument("--label-dir-name", default=DEFAULT_LABEL_DIR_NAME)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--patch-scale", type=float, default=0.2)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_CHOICES, default=None,
        help="Subset models cần test (mặc định: tất cả)"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Cần CUDA.")
    device = torch.device("cuda")

    # Dataset
    image_dir = DATASET_ROOT / args.split / "pos"
    label_dir = image_dir / args.label_dir_name
    if not label_dir.exists():
        raise FileNotFoundError(
            f"Label directory không tìm thấy: {label_dir}. "
            "Chạy prepare_inria_labels.py trước."
        )

    dataset = InriaDataset(
        str(image_dir), str(label_dir), max_lab=20, imgsize=args.image_size, shuffle=False
    )
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Patch
    patch_tensor = to_square_patch_tensor(args.patch, device)
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()

    # Output
    output_dir = args.output_dir or (
        MODERN_ATTACK_ROOT / "batch_outputs" / f"all_models_{args.patch.stem}_{args.split.lower()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_eval = args.models or list(MODEL_CHOICES)
    print(f"[eval_all] Đánh giá {len(models_to_eval)} models trên {len(dataset)} ảnh...")

    all_results = []
    for model_name in models_to_eval:
        print(f"  → {model_name}...", end="", flush=True)
        result = eval_single_model(
            model_name, patch_tensor, patch_transformer, patch_applier,
            dataloader, device, args,
        )
        all_results.append(result)
        if result.get("skipped"):
            print(f" [SKIP] {result.get('error', '')}")
        else:
            print(
                f" clean={result['mean_clean_max_conf']:.3f} "
                f"→ patched={result['mean_patched_max_conf']:.3f} "
                f"(↓{result['mean_conf_reduction']:.3f}) "
                f"fooled={result['fooled_rate_pct']:.1f}%"
            )

    # Lưu kết quả
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # In bảng tóm tắt
    print("\n" + "=" * 80)
    print(f"{'Model':<14} {'Clean':>8} {'Patched':>8} {'Reduce':>8} {'Fooled%':>9}")
    print("-" * 80)
    for r in all_results:
        if r.get("skipped"):
            print(f"{r['model']:<14} {'SKIP':>8}")
        else:
            print(
                f"{r['model']:<14} "
                f"{r['mean_clean_max_conf']:>8.3f} "
                f"{r['mean_patched_max_conf']:>8.3f} "
                f"{r['mean_conf_reduction']:>8.3f} "
                f"{r['fooled_rate_pct']:>8.1f}%"
            )
    print("=" * 80)
    print(f"\n[done] Kết quả lưu tại: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
