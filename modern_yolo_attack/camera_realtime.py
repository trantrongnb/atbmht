"""
Camera Real-time Demo – Naturalistic Modern YOLO Attack

Mở webcam, chạy YOLO detector trên mỗi frame, hiển thị:
  - Cửa sổ trái: ảnh gốc + detections (MÀU XANH)
  - Cửa sổ phải: ảnh đã dán patch + detections (MÀU ĐỎ nếu vẫn phát hiện)

Bấm 'q' để thoát.

Sử dụng:
  cd /home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack
  conda run -n signclip_310 python -m modern_yolo_attack.camera_realtime \\
    --model yolov8n \\
    --patch exp/yolov8n_biggan_ap/generated/patch-best.png

  # Hoặc với YOLOv10n:
  conda run -n signclip_310 python -m modern_yolo_attack.camera_realtime \\
    --model yolov10n \\
    --patch exp/yolov10n_biggan_ap/generated/patch-best.png

  # Chỉ detect (không dán patch):
  conda run -n signclip_310 python -m modern_yolo_attack.camera_realtime \\
    --model yolov8n --no-patch
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# Đảm bảo sys.path cho adversarialYolo & ultralytics
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
    load_modern_detector,
    to_square_patch_tensor,
)


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_boxes_cv2(
    frame: np.ndarray,
    result,
    conf_thres: float,
    cls_id: int = 0,
    color: tuple = (0, 255, 0),
    label_prefix: str = "",
) -> np.ndarray:
    """Vẽ bounding boxes lên OpenCV frame (BGR)."""
    out = frame.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return out

    h, w = out.shape[:2]
    for box in result.boxes:
        cid = int(box.cls.cpu().item())
        conf = float(box.conf.cpu().item())
        if cid != cls_id or conf < conf_thres:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        # Scale từ image_size về frame size nếu cần
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label_prefix}person {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw, y1), color, -1)
        cv2.putText(
            out, text,
            (x1, max(th + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return out


def extract_person_labels(
    result, img_size: int, cls_id: int = 0, max_labels: int = 14
) -> torch.Tensor | None:
    """
    Trích xuất labels YOLO [cls, x_c, y_c, w, h] normalized từ ultralytics result.
    Trả về tensor (1, N, 5) hoặc None.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None

    img_h, img_w = result.boxes.orig_shape
    labels = []
    for box in result.boxes:
        if int(box.cls.cpu().item()) != cls_id:
            continue
        conf = float(box.conf.cpu().item())
        x_c = float(box.xywh[0][0].cpu().item() / img_w)
        y_c = float(box.xywh[0][1].cpu().item() / img_h)
        bw = float(box.xywh[0][2].cpu().item() / img_w)
        bh = float(box.xywh[0][3].cpu().item() / img_h)
        labels.append([float(cls_id), x_c, y_c, bw, bh, conf])

    if not labels:
        return None

    labels_t = torch.tensor(labels, dtype=torch.float32)
    # Sắp xếp theo conf giảm dần, lấy top max_labels
    if labels_t.size(0) > max_labels:
        order = torch.argsort(labels_t[:, 5], descending=True)
        labels_t = labels_t[order[:max_labels]]

    return labels_t[:, :5].unsqueeze(0)  # (1, N, 5)


def count_persons(result, conf_thres: float, cls_id: int = 0) -> int:
    """Đếm số person detections trên ngưỡng."""
    if result.boxes is None:
        return 0
    count = 0
    for box in result.boxes:
        if int(box.cls.cpu().item()) == cls_id and float(box.conf.cpu().item()) >= conf_thres:
            count += 1
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Camera real-time demo: detect + adversarial patch overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=MODEL_CHOICES, default="yolov8n",
        help="YOLO model variant",
    )
    parser.add_argument(
        "--patch", type=Path, default=None,
        help="Đường dẫn tới patch image (.png). Bỏ trống hoặc dùng --no-patch để chỉ detect.",
    )
    parser.add_argument("--no-patch", action="store_true", help="Chỉ chạy detect, không dán patch")
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--image-size", type=int, default=416, help="Kích thước ảnh input cho YOLO")
    parser.add_argument("--patch-scale", type=float, default=0.2, help="Tỉ lệ patch so với bbox")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Ngưỡng confidence hiển thị")
    parser.add_argument("--camera-id", type=int, default=0, help="ID webcam (0, 1, ...)")
    parser.add_argument("--display-width", type=int, default=640, help="Chiều rộng cửa sổ hiển thị")
    args = parser.parse_args()

    use_patch = not args.no_patch and args.patch is not None
    if use_patch and not args.patch.exists():
        raise FileNotFoundError(f"Patch không tìm thấy: {args.patch}")

    if not torch.cuda.is_available():
        raise RuntimeError("Cần CUDA (PatchTransformer sử dụng CUDA tensors).")

    device = torch.device("cuda")
    to_tensor = T.ToTensor()

    # ── Load detector ──
    print(f"[init] Loading {args.model}...")
    detector, weights_path = load_modern_detector(args.model, device, args.weights_dir)
    print(f"[init] Weights: {weights_path}")

    # ── Load patch & transformers ──
    patch_tensor = None
    patch_transformer = None
    patch_applier = None
    if use_patch:
        print(f"[init] Loading patch: {args.patch}")
        patch_tensor = to_square_patch_tensor(args.patch, device)
        patch_transformer = PatchTransformer().cuda()
        patch_applier = PatchApplier().cuda()
        print(f"[init] Patch size: {patch_tensor.shape}")

    # ── Open camera ──
    print(f"[init] Opening camera {args.camera_id}...")
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở camera {args.camera_id}")

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[init] Camera resolution: {cam_w}x{cam_h}")

    mode_str = f"{args.model} | patch={'ON' if use_patch else 'OFF'} | conf≥{args.conf_thres}"
    print(f"\n[run] {mode_str}")
    print("[run] Bấm 'q' để thoát.\n")

    fps_list = []

    try:
        while True:
            t0 = time.time()
            ret, frame_bgr = cap.read()
            if not ret:
                print("[warn] Không đọc được frame, thử lại...")
                continue

            # ── Chuyển frame thành tensor ──
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = to_tensor(frame_rgb).unsqueeze(0)  # (1, 3, H, W)
            img_tensor = F.interpolate(img_tensor, size=args.image_size, mode="bilinear", align_corners=False)
            img_tensor = img_tensor.to(device)

            # ── Clean detection ──
            clean_result = detector(img_tensor, verbose=False, conf=args.conf_thres)[0]
            n_clean = count_persons(clean_result, args.conf_thres)

            # Resize clean_result frame để vẽ
            clean_display = F.interpolate(img_tensor, size=(args.display_width, args.display_width))[0]
            clean_np = (clean_display.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            clean_np = cv2.cvtColor(clean_np, cv2.COLOR_RGB2BGR)

            # Vẽ detections trên clean (chạy lại detect trên display size)
            clean_display_result = detector(
                F.interpolate(img_tensor, size=(args.display_width, args.display_width)),
                verbose=False, conf=args.conf_thres
            )[0]
            clean_vis = draw_boxes_cv2(clean_np, clean_display_result, args.conf_thres, color=(0, 255, 0))

            if use_patch:
                # ── Trích labels từ clean detection ──
                labels = extract_person_labels(clean_result, args.image_size)

                if labels is not None:
                    labels = labels.to(device)
                    # ── Dán patch lên ảnh ──
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
                    patched_tensor = patch_applier(img_tensor[0], adv_batch_t)

                    # ── Patched detection ──
                    patched_result = detector(patched_tensor, verbose=False, conf=args.conf_thres)[0]
                    n_patched = count_persons(patched_result, args.conf_thres)

                    # Display patched
                    patched_display = F.interpolate(
                        patched_tensor, size=(args.display_width, args.display_width)
                    )[0]
                    patched_np = (patched_display.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    patched_np = cv2.cvtColor(patched_np, cv2.COLOR_RGB2BGR)

                    patched_display_result = detector(
                        F.interpolate(patched_tensor, size=(args.display_width, args.display_width)),
                        verbose=False, conf=args.conf_thres
                    )[0]
                    patched_vis = draw_boxes_cv2(
                        patched_np, patched_display_result, args.conf_thres,
                        color=(0, 0, 255), label_prefix="[ATK] "
                    )
                else:
                    # Không detect được người → không dán patch
                    n_patched = 0
                    patched_vis = clean_np.copy()
                    cv2.putText(
                        patched_vis, "No person detected - patch not applied",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2,
                    )

                # ── FPS ──
                dt = time.time() - t0
                fps = 1.0 / max(dt, 1e-6)
                fps_list.append(fps)
                if len(fps_list) > 30:
                    fps_list.pop(0)
                avg_fps = sum(fps_list) / len(fps_list)

                # ── OSD text ──
                cv2.putText(
                    clean_vis, f"CLEAN | persons: {n_clean} | {avg_fps:.1f} FPS",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                cv2.putText(
                    patched_vis, f"PATCHED | persons: {n_patched} | fooled: {n_clean - n_patched}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )

                # ── Ghép 2 ảnh cạnh nhau ──
                combined = np.hstack([clean_vis, patched_vis])
                cv2.imshow(f"Naturalistic Attack - {args.model}", combined)

            else:
                # Chế độ chỉ detect (không dán patch)
                dt = time.time() - t0
                fps = 1.0 / max(dt, 1e-6)
                fps_list.append(fps)
                if len(fps_list) > 30:
                    fps_list.pop(0)
                avg_fps = sum(fps_list) / len(fps_list)

                cv2.putText(
                    clean_vis, f"{args.model} | persons: {n_clean} | {avg_fps:.1f} FPS",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                cv2.imshow(f"YOLO Detection - {args.model}", clean_vis)

            # ── Thoát khi bấm 'q' ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[stop] Ctrl+C")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[done] Camera đã đóng.")


if __name__ == "__main__":
    main()
