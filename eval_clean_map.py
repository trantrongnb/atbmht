"""
Đánh giá clean mAP (không có patch) của 3 model YOLO trên tập INRIA Train.
Dùng cùng pipeline DetectionValidator như train script để đảm bảo tính nhất quán.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
for _candidate in (
    PROJECT_ROOT,
    PROJECT_ROOT / "adversarialYolo",
    PROJECT_ROOT / "GANLatentDiscovery",
    PROJECT_ROOT / "ultralytics",
):
    _candidate_str = str(_candidate)
    if _candidate.exists() and _candidate_str not in sys.path:
        sys.path.insert(0, _candidate_str)

from adversarialYolo.load_data import InriaDataset
from modern_yolo_attack.bridge import bootstrap_paths
from modern_yolo_attack.common import (
    DATASET_ROOT,
    DEFAULT_LABEL_DIR_NAME,
    DEFAULT_WEIGHTS_DIR,
    load_modern_detector,
)
from modern_yolo_attack.train_biggan_modern_patch import (
    build_ap_validator,
    lab_batch_to_validator_batch,
)


def evaluate_clean_map(
    model_name: str,
    device: torch.device,
    image_size: int = 416,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.6,
    max_det: int = 300,
) -> dict:
    """Tính mAP trên ảnh sạch (không dán patch)."""
    detector, weights_path = load_modern_detector(model_name, device, DEFAULT_WEIGHTS_DIR)

    split = "Train"
    image_dir = DATASET_ROOT / split / "pos"
    label_dir = image_dir / DEFAULT_LABEL_DIR_NAME

    dataset = InriaDataset(
        str(image_dir), str(label_dir), max_lab=20, imgsize=image_size, shuffle=False
    )
    dataset = torch.utils.data.Subset(dataset, list(range(min(64, len(dataset)))))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    save_dir = Path(f"eval_clean/{model_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    validator = build_ap_validator(
        detector=detector,
        image_size=image_size,
        save_dir=save_dir,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_det=max_det,
    )
    detector.model.eval()

    with torch.no_grad():
        for img_batch, lab_batch in dataloader:
            img_batch = img_batch.to(device)
            lab_batch = lab_batch.to(device)
            # Clean inference - không dán patch
            preds = validator.postprocess(detector.model(img_batch))
            validator_batch = lab_batch_to_validator_batch(img_batch, lab_batch)
            validator.update_metrics(preds, validator_batch)

    stats = validator.get_stats()
    validator.finalize_metrics()

    instances = (
        int(validator.nt_per_class.sum())
        if validator.nt_per_class is not None
        else 0
    )

    result = {
        "model": model_name,
        "weights": str(weights_path),
        "precision": float(stats.get("metrics/precision(B)", 0.0)),
        "recall": float(stats.get("metrics/recall(B)", 0.0)),
        "ap50": float(stats.get("metrics/mAP50(B)", 0.0)),
        "ap50_95": float(stats.get("metrics/mAP50-95(B)", 0.0)),
        "images": int(validator.seen or 0),
        "instances": instances,
    }
    return result


def main():
    device = torch.device("cuda")
    models = ["yolov8n", "yolov9n", "yolov10n"]
    results = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Evaluating clean mAP: {model_name}")
        print(f"{'='*60}")
        result = evaluate_clean_map(model_name, device)
        results[model_name] = result
        print(json.dumps(result, indent=2))

    output_path = Path("eval_clean/clean_map_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
