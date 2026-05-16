from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
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

from adversarialYolo.load_data import PatchApplier, PatchTransformer

from .common import (
    DEFAULT_WEIGHTS_DIR,
    MODEL_CHOICES,
    MODERN_ATTACK_ROOT,
    draw_ultralytics_detections,
    image_to_tensor,
    load_modern_detector,
    person_labels_from_result,
    summarize_person_result,
    to_square_patch_tensor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a patch to one image and test a modern YOLO detector.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="yolov8n")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--patch-scale", type=float, default=0.2)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This demo requires CUDA because PatchTransformer uses CUDA tensors internally.")

    device = torch.device("cuda")
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
    detector, weights_path = load_modern_detector(args.model, device, args.weights_dir)

    _, clean_tensor = image_to_tensor(args.image, args.image_size, device)
    patch_tensor = to_square_patch_tensor(args.patch, device)
    output_dir = args.output_dir or (MODERN_ATTACK_ROOT / "demo_outputs" / f"{args.model}_{args.patch.stem}_{args.image.stem}")
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_result = detector(clean_tensor, verbose=False, conf=args.conf_thres)[0]
    labels_tensor = person_labels_from_result(clean_result)
    if labels_tensor is None:
        raise RuntimeError(f"No clean person detection found in {args.image}")
    labels_tensor = labels_tensor.to(device)

    adv_batch_t, _, _ = patch_transformer(
        adv_patch=patch_tensor[0],
        lab_batch=labels_tensor,
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
    patched_tensor = patch_applier(clean_tensor[0], adv_batch_t)
    patched_result = detector(patched_tensor, verbose=False, conf=args.conf_thres)[0]

    clean_summary = summarize_person_result(clean_result, args.conf_thres)
    patched_summary = summarize_person_result(patched_result, args.conf_thres)
    score_reduction = clean_summary["max_person_conf"] - patched_summary["max_person_conf"]

    save_image(clean_tensor, output_dir / "clean_resized.png")
    save_image(patched_tensor, output_dir / "patched_applied.png")
    draw_ultralytics_detections(clean_tensor[0], clean_result, detector.names, args.conf_thres).save(
        output_dir / "clean_detected.png"
    )
    draw_ultralytics_detections(patched_tensor[0], patched_result, detector.names, args.conf_thres).save(
        output_dir / "patched_detected.png"
    )

    summary = {
        "model": args.model,
        "weights": str(weights_path),
        "image": str(args.image),
        "patch": str(args.patch),
        "image_size": args.image_size,
        "patch_scale": args.patch_scale,
        "conf_threshold": args.conf_thres,
        "clean": clean_summary,
        "patched": patched_summary,
        "score_reduction": score_reduction,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
