"""
Đánh giá so sánh với phương pháp SOTA (Patch_tot_nhat).
Test 3 patch-best của 3 model trên chính model đó ở 2 scale 0.22 và 0.25.
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
SCALES = [0.2, 0.22, 0.25]


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

    save_dir = Path(f"eval_sota/{model_name}_scale_{patch_scale}")
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

    # ── SOTA comparison evaluation ──
    all_results = {}
    
    for scale in SCALES:
        for detector_name in MODELS:
            patch_path = PROJECT_ROOT / "exp" / f"{detector_name}_biggan_ap" / "generated" / "patch-best.png"
            eval_key = f"{detector_name}_scale_{scale}"
            print(f"\n{'='*60}")
            print(f"  Evaluating Model: {detector_name} with Scale: {scale}")
            print(f"{'='*60}")
            result = evaluate_patched_map(detector_name, patch_path, device, patch_scale=scale)
            all_results[eval_key] = result
            print(f"mAP50 (patched) = {result['ap50']:.4f}")

    # ── Save raw results ──
    output_dir = PROJECT_ROOT / "eval_sota"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "sota_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Update so_sanh.md ──
    update_comparison_md(clean_results, all_results)
    print(f"\n[DONE] Results saved. so_sanh.md updated.")


def update_comparison_md(clean_results: dict, sota_results: dict):
    # Dữ liệu phương pháp khác (từ file so_sanh_voi_sota.txt và bảng 2)
    other_method = {
        "yolov8n": {0.2: 31.27, 0.22: 31.27, 0.25: 23.40},
        "yolov9n": {0.2: 41.30, 0.22: 41.30, 0.25: 39.97},
        "yolov10n": {0.2: 24.51, 0.22: 24.51, 0.25: 28.62},
    }
    
    # Đọc lại nội dung cũ của so_sanh.md (nếu có)
    md_path = PROJECT_ROOT / "so_sanh.md"
    content = ""
    if md_path.exists():
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

    # Thêm bảng so sánh mới
    lines = []
    lines.append("\n---\n")
    lines.append("## 3. Đánh giá và So sánh với phương pháp SOTA (Patch_tot_nhat) ở các Scale 0.2, 0.22 và 0.25\n")
    lines.append("Kết quả đánh giá trên tập INRIA Train, so sánh trực tiếp mAP còn lại (%) của hệ thống giữa 2 phương pháp.\n")

    lines.append("| Mô hình | Clean mAP50 | Our (0.2) | SOTA (0.2) | Our (0.22) | SOTA (0.22) | Our (0.25) | SOTA (0.25) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m in MODELS:
        c50 = clean_results.get(m, {}).get("ap50", 0) * 100
        
        our_02 = sota_results.get(f"{m}_scale_0.2", {}).get("ap50", 0) * 100
        our_022 = sota_results.get(f"{m}_scale_0.22", {}).get("ap50", 0) * 100
        our_025 = sota_results.get(f"{m}_scale_0.25", {}).get("ap50", 0) * 100
        
        om_02 = other_method[m][0.2]
        om_022 = other_method[m][0.22]
        om_025 = other_method[m][0.25]
        
        # Bôi đậm nếu phương pháp của mình tốt hơn (mAP còn lại thấp hơn)
        our_02_str = f"**{our_02:.2f}%**" if our_02 < om_02 else f"{our_02:.2f}%"
        om_02_str = f"**{om_02:.2f}%**" if om_02 < our_02 else f"{om_02:.2f}%"

        our_022_str = f"**{our_022:.2f}%**" if our_022 < om_022 else f"{our_022:.2f}%"
        om_022_str = f"**{om_022:.2f}%**" if om_022 < our_022 else f"{om_022:.2f}%"
        
        our_025_str = f"**{our_025:.2f}%**" if our_025 < om_025 else f"{our_025:.2f}%"
        om_025_str = f"**{om_025:.2f}%**" if om_025 < our_025 else f"{om_025:.2f}%"

        lines.append(
            f"| **{m}** "
            f"| {c50:.2f}% "
            f"| {our_02_str} "
            f"| {om_02_str} "
            f"| {our_022_str} "
            f"| {om_022_str} "
            f"| {our_025_str} "
            f"| {om_025_str} |"
        )
    
    lines.append("\n*Nhận xét:* Tỷ lệ mAP càng thấp chứng tỏ khả năng tấn công càng mạnh. Các con số in đậm thể hiện phương pháp có khả năng qua mặt (fool) mô hình tốt hơn tại cùng kích thước bản vá.\n")

    # Append to existing content or write new if doesn't exist
    if "## 3. Đánh giá và So sánh với phương pháp SOTA" in content:
        # Thay thế phần cũ nếu đã ghi trước đó
        parts = content.split("## 3. Đánh giá và So sánh với phương pháp SOTA")
        content = parts[0].strip()
        
    final_content = content + "\n" + "\n".join(lines)
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(final_content)


if __name__ == "__main__":
    main()
