"""
Đánh giá Transfer Attack: Patch được train trên model A → test trên tất cả model khác.
Tạo ma trận transferability (source x target).
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


def compute_fooled_rate(
    detector,
    patch_tensor: torch.Tensor,
    patch_transformer: PatchTransformer,
    patch_applier: PatchApplier,
    dataloader: DataLoader,
    device: torch.device,
    conf_thres: float,
    patch_scale: float,
    image_size: int,
) -> dict:
    """Tính tỉ lệ fooled và mean confidence reduction."""
    total_clean_conf = 0.0
    total_patched_conf = 0.0
    n_fooled = 0
    n_images = 0

    for clean_imgs, labels in dataloader:
        clean_imgs = clean_imgs.to(device)
        labels = labels.to(device)

        clean_results = detector(clean_imgs, verbose=False, conf=conf_thres)

        adv_batch_t, _, _ = patch_transformer(
            adv_patch=patch_tensor[0],
            lab_batch=labels,
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
            enable_empty_patch=False,
            enable_no_random=True,
            enable_blurred=False,
        )
        patched_imgs = patch_applier(clean_imgs, adv_batch_t)
        patched_results = detector(patched_imgs, verbose=False, conf=conf_thres)

        for i in range(clean_imgs.size(0)):
            c = summarize_person_result(clean_results[i], conf_thres)
            p = summarize_person_result(patched_results[i], conf_thres)
            total_clean_conf += c["max_person_conf"]
            total_patched_conf += p["max_person_conf"]
            if p["person_count_above_threshold"] == 0:
                n_fooled += 1
            n_images += 1

    n = max(n_images, 1)
    return {
        "mean_clean_conf": round(total_clean_conf / n, 4),
        "mean_patched_conf": round(total_patched_conf / n, 4),
        "mean_reduction": round((total_clean_conf - total_patched_conf) / n, 4),
        "fooled_rate_pct": round(n_fooled / n * 100.0, 2),
        "n_images": n_images,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer attack evaluation: test patch trên nhiều models,\n"
            "tạo ma trận transferability."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--patches", nargs="+", type=Path, required=True,
        help=(
            "Các cặp source_model:patch_path, VD:\n"
            "  yolov8n:/path/to/patch_v8n.png yolov9t:/path/to/patch_v9t.png\n"
            "Hoặc chỉ một patch file để test với tất cả targets."
        ),
    )
    parser.add_argument(
        "--source-models", nargs="+", choices=MODEL_CHOICES, default=None,
        help="Tên source models (theo thứ tự với --patches)"
    )
    parser.add_argument(
        "--target-models", nargs="+", choices=MODEL_CHOICES, default=None,
        help="Models cần test (mặc định: tất cả MODEL_CHOICES)"
    )
    parser.add_argument("--split", choices=("Train", "Test"), default="Test")
    parser.add_argument("--label-dir-name", default=DEFAULT_LABEL_DIR_NAME)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--patch-scale", type=float, default=0.2)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Cần CUDA.")
    device = torch.device("cuda")

    # Dataset
    image_dir = DATASET_ROOT / args.split / "pos"
    label_dir = image_dir / args.label_dir_name
    dataset = InriaDataset(
        str(image_dir), str(label_dir), max_lab=20, imgsize=args.image_size, shuffle=False
    )
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()

    # Xác định patch sources
    patch_paths = args.patches
    source_names = args.source_models or [f"patch_{i}" for i in range(len(patch_paths))]
    target_models = args.target_models or list(MODEL_CHOICES)

    # Output
    output_dir = args.output_dir or (MODERN_ATTACK_ROOT / "batch_outputs" / "transferability")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ma trận kết quả: {source_name: {target_model: metrics}}
    matrix: dict[str, dict[str, dict]] = {}

    for source_name, patch_path in zip(source_names, patch_paths):
        if not patch_path.exists():
            print(f"[SKIP] {source_name}: patch không tồn tại {patch_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Source: {source_name} | Patch: {patch_path.name}")
        print("=" * 60)

        patch_tensor = to_square_patch_tensor(patch_path, device)
        matrix[source_name] = {}

        for target_model in target_models:
            print(f"  → Target: {target_model}...", end="", flush=True)
            try:
                detector, _ = load_modern_detector(target_model, device, args.weights_dir)
                metrics = compute_fooled_rate(
                    detector, patch_tensor, patch_transformer, patch_applier,
                    dataloader, device,
                    conf_thres=args.conf_thres,
                    patch_scale=args.patch_scale,
                    image_size=args.image_size,
                )
                matrix[source_name][target_model] = metrics
                print(
                    f" clean={metrics['mean_clean_conf']:.3f}"
                    f" patched={metrics['mean_patched_conf']:.3f}"
                    f" fooled={metrics['fooled_rate_pct']:.1f}%"
                )
            except Exception as e:
                matrix[source_name][target_model] = {"error": str(e)}
                print(f" [ERROR] {e}")

    # Lưu kết quả
    results_path = output_dir / "transferability_matrix.json"
    with open(results_path, "w") as f:
        json.dump(matrix, f, indent=2)

    # In ma trận fooled_rate
    print("\n\n" + "=" * 90)
    print("TRANSFERABILITY MATRIX – Fooled Rate (%) [source → target]")
    print("=" * 90)
    col_w = 10
    header = f"{'Source\\Target':<16}" + "".join(f"{m:>{col_w}}" for m in target_models)
    print(header)
    print("-" * 90)
    for src, targets in matrix.items():
        row = f"{src:<16}"
        for tgt in target_models:
            val = targets.get(tgt, {})
            if "error" in val:
                row += f"{'ERR':>{col_w}}"
            else:
                row += f"{val.get('fooled_rate_pct', 0.0):>{col_w}.1f}"
        print(row)
    print("=" * 90)

    print(f"\n[done] Ma trận đã lưu: {results_path}")


if __name__ == "__main__":
    main()
