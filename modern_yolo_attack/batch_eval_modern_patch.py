from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

from .common import (
    DATASET_ROOT,
    DEFAULT_LABEL_DIR_NAME,
    DEFAULT_WEIGHTS_DIR,
    MODEL_CHOICES,
    MODERN_ATTACK_ROOT,
    draw_ultralytics_detections,
    load_modern_detector,
    summarize_person_result,
    to_square_patch_tensor,
)


def build_patch_tensor(mode: str, patch_path: Path | None, device: torch.device, image_size: int) -> tuple[torch.Tensor, str]:
    if mode == "learned":
        if patch_path is None or not patch_path.exists():
            raise FileNotFoundError(f"Patch not found: {patch_path}")
        return to_square_patch_tensor(patch_path, device), patch_path.stem
    if mode == "black":
        return torch.zeros((1, 3, image_size, image_size), device=device, dtype=torch.float32), "black"
    if mode == "white":
        return torch.ones((1, 3, image_size, image_size), device=device, dtype=torch.float32), "white"
    if mode == "gray":
        return torch.full((1, 3, image_size, image_size), 0.5, device=device, dtype=torch.float32), "gray"
    raise ValueError(f"Unsupported patch mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate a patch against a modern YOLO model on INRIA.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="yolov8n")
    parser.add_argument("--split", choices=("Train", "Test"), default="Train")
    parser.add_argument("--patch", type=Path, default=None)
    parser.add_argument("--baseline", choices=("learned", "black", "white", "gray"), default="learned")
    parser.add_argument("--label-dir-name", default=DEFAULT_LABEL_DIR_NAME)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--patch-scale", type=float, default=0.2)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--save-images", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This batch evaluation requires CUDA because PatchTransformer uses CUDA tensors internally.")

    device = torch.device("cuda")
    image_dir = DATASET_ROOT / args.split / "pos"
    label_dir = image_dir / args.label_dir_name
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}. Run prepare_inria_labels.py first.")

    patch_tensor, patch_label = build_patch_tensor(args.baseline, args.patch, device, args.image_size)
    detector, weights_path = load_modern_detector(args.model, device, args.weights_dir)
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()

    dataset = InriaDataset(str(image_dir), str(label_dir), max_lab=20, imgsize=args.image_size, shuffle=False)
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    output_dir = args.output_dir or (
        MODERN_ATTACK_ROOT / "batch_outputs" / f"{args.model}_{patch_label}_{args.split.lower()}_{len(dataset)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    results = []
    saved = 0
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])[: len(dataset)]

    image_offset = 0
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

        for batch_index in range(clean_imgs.size(0)):
            image_path = image_paths[image_offset + batch_index]
            clean_summary = summarize_person_result(clean_results[batch_index], args.conf_thres)
            patched_summary = summarize_person_result(patched_results[batch_index], args.conf_thres)
            score_reduction = clean_summary["max_person_conf"] - patched_summary["max_person_conf"]
            results.append(
                {
                    "image": str(image_path),
                    "clean": clean_summary,
                    "patched": patched_summary,
                    "score_reduction": score_reduction,
                    "fooled_zero_person": patched_summary["person_count_above_threshold"] == 0,
                }
            )

            if saved < args.save_images:
                sample_dir = samples_dir / image_path.stem
                sample_dir.mkdir(parents=True, exist_ok=True)
                save_image(clean_imgs[batch_index], sample_dir / "clean_resized.png")
                save_image(patched_imgs[batch_index], sample_dir / "patched_applied.png")
                draw_ultralytics_detections(
                    clean_imgs[batch_index], clean_results[batch_index], detector.names, args.conf_thres
                ).save(sample_dir / "clean_detected.png")
                draw_ultralytics_detections(
                    patched_imgs[batch_index], patched_results[batch_index], detector.names, args.conf_thres
                ).save(sample_dir / "patched_detected.png")
                saved += 1

        image_offset += clean_imgs.size(0)

    reduced = [r for r in results if r["score_reduction"] > 0]
    fooled = [r for r in results if r["fooled_zero_person"]]
    summary = {
        "model": args.model,
        "baseline": args.baseline,
        "patch": str(args.patch) if args.baseline == "learned" else patch_label,
        "weights": str(weights_path),
        "split": args.split,
        "label_dir": str(label_dir),
        "num_images_tested": len(results),
        "num_images_reduced_score": len(reduced),
        "num_images_fooled_zero_person": len(fooled),
        "mean_clean_max_person_score": sum(r["clean"]["max_person_conf"] for r in results) / len(results) if results else 0.0,
        "mean_patched_max_person_score": sum(r["patched"]["max_person_conf"] for r in results) / len(results) if results else 0.0,
        "mean_score_reduction": sum(r["score_reduction"] for r in results) / len(results) if results else 0.0,
        "output_dir": str(output_dir),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "per_image_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
