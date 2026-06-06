"""
Đánh giá cross-model mAP: Dùng 6 patches (best và last của v8, v9, v10) test trên tất cả 3 model.
Kết quả tạo ra ma trận 3x6 (Detector × Patch).
Ghi kết quả tính toán vào file so_sanh.md.
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

from adversarialYolo.load_data import InriaDataset, PatchApplier, PatchTransformer

from modern_yolo_attack.bridge import bootstrap_paths
from modern_yolo_attack.common import (
    DATASET_ROOT,
    DEFAULT_LABEL_DIR_NAME,
    DEFAULT_WEIGHTS_DIR,
    load_modern_detector,
    to_square_patch_tensor,
)
from modern_yolo_attack.train_biggan_modern_patch import (
    build_ap_validator,
    lab_batch_to_validator_batch,
    reset_detector_inference_cache,
)


MODELS = ["yolov8n", "yolov9n", "yolov10n"]

# Define 6 patches
PATCH_MAP = {
    "v8n_best": PROJECT_ROOT / "exp" / "yolov8n_biggan_ap" / "generated" / "patch-best.png",
    "v8n_last": PROJECT_ROOT / "exp" / "yolov8n_biggan_ap" / "generated" / "patch-last.png",
    "v9n_best": PROJECT_ROOT / "exp" / "yolov9n_biggan_ap" / "generated" / "patch-best.png",
    "v9n_last": PROJECT_ROOT / "exp" / "yolov9n_biggan_ap" / "generated" / "patch-last.png",
    "v10n_best": PROJECT_ROOT / "exp" / "yolov10n_biggan_ap" / "generated" / "patch-best.png",
    "v10n_last": PROJECT_ROOT / "exp" / "yolov10n_biggan_ap" / "generated" / "patch-last.png",
}


def evaluate_patched_map(
    model_name: str,
    patch_path: Path,
    device: torch.device,
    image_size: int = 416,
    patch_scale: float = 0.2,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.6,
    max_det: int = 300,
) -> dict:
    """Tính mAP trên ảnh đã dán patch."""
    detector, weights_path = load_modern_detector(model_name, device, DEFAULT_WEIGHTS_DIR)
    patch_tensor = to_square_patch_tensor(patch_path, device)
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()

    split = "Train"
    image_dir = DATASET_ROOT / split / "pos"
    label_dir = image_dir / DEFAULT_LABEL_DIR_NAME

    dataset = InriaDataset(
        str(image_dir), str(label_dir), max_lab=20, imgsize=image_size, shuffle=False
    )
    dataset = torch.utils.data.Subset(dataset, list(range(min(64, len(dataset)))))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    save_dir = Path(f"eval_cross/{model_name}_{patch_path.parent.parent.name}_{patch_path.stem}")
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

            adv_batch_t, _, _ = patch_transformer(
                adv_patch=patch_tensor[0],
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
    instances = (
        int(validator.nt_per_class.sum())
        if validator.nt_per_class is not None
        else 0
    )

    reset_detector_inference_cache(detector)

    return {
        "ap50": float(stats.get("metrics/mAP50(B)", 0.0)),
    }


def main():
    device = torch.device("cuda")

    # ── Load clean baseline ──
    clean_path = PROJECT_ROOT / "eval_clean" / "clean_map_results.json"
    if clean_path.exists():
        with open(clean_path) as f:
            clean_results = json.load(f)
    else:
        print("[ERROR] clean_map_results.json not found! Run eval_clean_map.py first.")
        return

    # ── Cross-model evaluation ──
    all_results = {}
    patch_keys = list(PATCH_MAP.keys())
    
    for detector_name in MODELS:
        for patch_key in patch_keys:
            patch_path = PATCH_MAP[patch_key]
            eval_key = f"{detector_name}__{patch_key}"
            print(f"\n{'='*60}")
            print(f"  Evaluating Model: {detector_name} with Patch: {patch_key}")
            print(f"{'='*60}")
            result = evaluate_patched_map(detector_name, patch_path, device)
            all_results[eval_key] = result
            print(f"mAP50 (patched) = {result['ap50']:.4f}")

    # ── Save raw results ──
    output_dir = PROJECT_ROOT / "eval_cross"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "cross_model_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Build so_sanh.md ──
    build_comparison_md(clean_results, all_results, patch_keys)
    print(f"\n[DONE] Results saved. so_sanh.md updated.")


def build_comparison_md(clean_results: dict, cross_results: dict, patch_keys: list[str]):
    lines = []
    lines.append("# Đánh giá hiệu năng tấn công Adversarial Patch\n")
    
    # ── BẢNG 1: Ma trận sụt giảm mAP ──
    lines.append("## 1. Bảng đánh giá độ sụt giảm mAP của 3 mô hình trên 6 patch")
    lines.append("Ma trận thể hiện điểm sụt giảm mAP50 (Clean mAP50 - Patched mAP50).\n")
    
    header = "| Mô hình \\ Patch |"
    for pk in patch_keys:
        header += f" {pk} |"
    lines.append(header)
    
    align = "| :--- |" + " :---: |" * len(patch_keys)
    lines.append(align)

    for det in MODELS:
        row = f"| **{det}** |"
        clean_ap50 = clean_results.get(det, {}).get("ap50", 0)
        
        for pk in patch_keys:
            eval_key = f"{det}__{pk}"
            patched_ap50 = cross_results.get(eval_key, {}).get("ap50", 0)
            drop = (clean_ap50 - patched_ap50) * 100
            row += f" **-{(drop):.2f}%** |"
            
        lines.append(row)
    lines.append("\n")

    # ── BẢNG 2: So sánh với phương pháp khác ──
    lines.append("## 2. So sánh với phương pháp khác (Patch_tot nhat)\n")
    lines.append("Dưới đây là bảng đánh giá hiệu năng mAP của các mô hình YOLO bị tấn công bởi phương pháp khác (Ghi chú: Các số liệu thể hiện **mAP còn lại**).\n")

    lines.append("| Mô hình | Scale 0.2 (Bản vá lớn hơn - Bảng 9) <br> Tấn công bởi **Patch_tot nhat** | Scale 0.25 (Bản vá rất lớn - Bảng 5) <br> Tấn công bởi **Patch_tot nhat_0.25** |")
    lines.append("| :--- | :---: | :---: |")
    lines.append("| **YOLOv8n** | **31.27%** | **24.77%** |")
    lines.append("| **YOLOv9n** | **41.30%** | **33.13%** |")
    lines.append("| **YOLOv10n**| **24.51%** *(Sụt giảm mạnh)* | **14.94%** *(Chạm đáy - Bị \"mù\")* |")
    lines.append("\n*Nhận xét:* Ở kích thước bản vá $0.25$, hệ thống nhận diện của cả 3 mô hình gần như bị vô hiệu hóa (\"mù\" hoàn toàn), trong đó **YOLOv10n** thể hiện sự kém chống chịu nhất trước các dạng tấn công dạng bản vá (Adversarial Patch)..\n")

    md_content = "\n".join(lines)
    with open(PROJECT_ROOT / "so_sanh.md", "w", encoding="utf-8") as f:
        f.write(md_content)

if __name__ == "__main__":
    main()
