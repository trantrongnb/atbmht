from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.request import urlretrieve

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

from .bridge import (
    MODERN_SOURCE_REPO,
    OLD_REPO_ROOT,
    PROJECT_ROOT,
    bootstrap_paths,
    import_yolo_class,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
MODERN_ATTACK_ROOT = PACKAGE_ROOT.parent
LOCAL_DATASET_ROOT = PROJECT_ROOT / "dataset" / "inria"
DATASET_ROOT = LOCAL_DATASET_ROOT if LOCAL_DATASET_ROOT.exists() else OLD_REPO_ROOT / "dataset" / "inria"
DEFAULT_LABEL_DIR_NAME = "yolo-labels_modern_gt"
DEFAULT_WEIGHTS_DIR = MODERN_ATTACK_ROOT / "weights"
LEGACY_WEIGHTS_DIR = OLD_REPO_ROOT / "modern_yolo_attack" / "weights"

# ──────────────────────────────────────────────────────────────────────────────
# Chỉ tập trung vào YOLOv8 / YOLOv9 / YOLOv10
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CHOICES = (
    # YOLOv8
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    # YOLOv9
    "yolov9n",
    "yolov9t",
    "yolov9s",
    "yolov9m",
    "yolov9c",
    # YOLOv10
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov10l",
)

MODEL_ASSET_MAP = {
    # YOLOv8
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    # YOLOv9 (ultralytics-packaged)
    "yolov9n": "yolov9t.pt",
    "yolov9t": "yolov9t.pt",
    "yolov9s": "yolov9s.pt",
    "yolov9m": "yolov9m.pt",
    "yolov9c": "yolov9c.pt",
    # YOLOv10
    "yolov10n": "yolov10n.pt",
    "yolov10s": "yolov10s.pt",
    "yolov10m": "yolov10m.pt",
    "yolov10l": "yolov10l.pt",
}

# URL download ultralytics assets – v8.2.0 chứa đầy đủ v8/v9/v10
ASSET_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def resolve_model_asset_name(model_name: str) -> str:
    if model_name not in MODEL_ASSET_MAP:
        raise ValueError(
            f"Unsupported model: '{model_name}'. Supported: {MODEL_CHOICES}"
        )
    return MODEL_ASSET_MAP[model_name]


def ensure_model_weights(
    model_name: str, weights_dir: Path = DEFAULT_WEIGHTS_DIR
) -> Path:
    """
    Đảm bảo YOLO weights tồn tại: tìm kiếm local trước, sau đó auto-download.
    """
    weights_dir.mkdir(parents=True, exist_ok=True)
    asset_name = resolve_model_asset_name(model_name)
    local_path = weights_dir / asset_name
    if local_path.exists():
        return local_path

    # Kiểm tra các nguồn local khác
    candidate_sources = (
        LEGACY_WEIGHTS_DIR / asset_name,
        MODERN_SOURCE_REPO / "weights" / asset_name,
    )
    for source_path in candidate_sources:
        if source_path.exists():
            shutil.copy2(source_path, local_path)
            return local_path

    # Auto-download từ ultralytics github releases
    print(f"[weights] Đang tải {asset_name} từ ultralytics assets...")
    urlretrieve(f"{ASSET_BASE_URL}/{asset_name}", local_path)
    return local_path


def load_modern_detector(
    model_name: str,
    device: torch.device,
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
):
    """Load YOLO detector (YOLOv8/v9/v10) lên device."""
    YOLO = import_yolo_class()
    weights_path = ensure_model_weights(model_name, weights_dir)
    detector = YOLO(str(weights_path))
    detector.to(device)
    detector.model.eval()
    for param in detector.model.parameters():
        param.requires_grad_(False)
    return detector, weights_path


def load_multiple_detectors(
    model_names: list[str],
    device: torch.device,
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
) -> list:
    """
    Load nhiều YOLO detectors cùng lúc cho ensemble attack.
    Trả về list [(detector, weights_path), ...]
    """
    detectors = []
    for model_name in model_names:
        det, wpath = load_modern_detector(model_name, device, weights_dir)
        detectors.append((det, wpath))
        print(f"  [OK] Loaded {model_name} từ {wpath.name}")
    return detectors


def build_attack_detection_criterion(detector):
    """
    Tạo criterion khả vi cho adversarial patch training.

    Dùng v8DetectionLoss cho cả YOLOv8/v9/v10. Với YOLOv10, training loop sẽ
    truyền riêng nhánh `one2many`, là phần còn gradient về input/patch.
    """
    bootstrap_paths()
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import v8DetectionLoss

    model = detector.model
    model.args = get_cfg(overrides=model.args)
    model.end2end = getattr(model, "end2end", getattr(model.model[-1], "end2end", False))
    return v8DetectionLoss(model)


def to_square_patch_tensor(
    image_path: Path, device: torch.device
) -> torch.Tensor:
    """Load patch image và chuyển thành tensor vuông (1, 3, H, H)."""
    patch = Image.open(image_path).convert("RGB")
    side = max(patch.size)
    patch = patch.resize((side, side))
    return T.ToTensor()(patch).unsqueeze(0).to(device, torch.float32)


def image_to_tensor(
    image_path: Path, image_size: int, device: torch.device
) -> tuple[Image.Image, torch.Tensor]:
    """Load và resize ảnh, trả về (PIL, tensor)."""
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    tensor = T.ToTensor()(image).unsqueeze(0).to(device, torch.float32)
    return image, tensor


def summarize_person_result(
    result, conf_threshold: float = 0.25, cls_id: int = 0
) -> dict:
    """Tổng hợp kết quả detections (class=person) từ một result của ultralytics."""
    if result.boxes is None or len(result.boxes) == 0:
        return {
            "person_count_above_threshold": 0,
            "max_person_conf": 0.0,
            "mean_person_conf": 0.0,
        }

    scores = []
    for box in result.boxes:
        if int(box.cls.cpu().item()) != cls_id:
            continue
        conf = float(box.conf.cpu().item())
        if conf >= conf_threshold:
            scores.append(conf)

    if not scores:
        return {
            "person_count_above_threshold": 0,
            "max_person_conf": 0.0,
            "mean_person_conf": 0.0,
        }

    return {
        "person_count_above_threshold": len(scores),
        "max_person_conf": max(scores),
        "mean_person_conf": float(sum(scores) / len(scores)),
    }


def person_labels_from_result(
    result, max_labels: int = 20, cls_id: int = 0
) -> torch.Tensor | None:
    """
    Trích xuất labels dạng YOLO từ ultralytics result.
    Output shape: (1, max_labels, 5) với 5 = [cls, x_c, y_c, w, h] normalized.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None

    img_h, img_w = result.boxes.orig_shape
    labels = []
    for box in result.boxes:
        if int(box.cls.cpu().item()) != cls_id:
            continue
        labels.append(
            [
                float(cls_id),
                float(box.xywh[0][0].cpu().item() / img_w),
                float(box.xywh[0][1].cpu().item() / img_h),
                float(box.xywh[0][2].cpu().item() / img_w),
                float(box.xywh[0][3].cpu().item() / img_h),
                float(box.conf.cpu().item()),
            ]
        )

    if not labels:
        return None

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    if labels_tensor.size(0) > max_labels:
        order = torch.argsort(labels_tensor[:, 5], descending=True)
        labels_tensor = labels_tensor[order[:max_labels]]
    return labels_tensor[:, :5].unsqueeze(0)


def draw_ultralytics_detections(
    image_tensor: torch.Tensor,
    result,
    names: dict,
    conf_threshold: float = 0.25,
    cls_id: int = 0,
) -> Image.Image:
    """Vẽ bounding boxes lên ảnh từ ultralytics result."""
    image = T.ToPILImage()(image_tensor.detach().cpu())
    draw = ImageDraw.Draw(image)
    if result.boxes is None or len(result.boxes) == 0:
        return image

    for box in result.boxes:
        current_cls_id = int(box.cls.cpu().item())
        conf = float(box.conf.cpu().item())
        if current_cls_id != cls_id or conf < conf_threshold:
            continue

        left, top, right, bottom = (
            int(box.xyxy[0][0].cpu().item()),
            int(box.xyxy[0][1].cpu().item()),
            int(box.xyxy[0][2].cpu().item()),
            int(box.xyxy[0][3].cpu().item()),
        )
        label = names.get(current_cls_id, str(current_cls_id))
        draw.rectangle([left, top, right, bottom], outline="red", width=2)
        draw.text((left, max(0, top - 12)), f"{label} {conf:.2f}", fill="red")
    return image


def parse_inria_annotation(
    annotation_path: Path,
) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]]:
    """Parse INRIA annotation .txt file → (image_size, list_of_boxes)."""
    text = annotation_path.read_text(errors="ignore")
    size_match = re.search(
        r"Image size \(X x Y x C\) : (\d+) x (\d+) x \d+", text
    )
    if size_match is None:
        raise ValueError(f"Could not parse image size from {annotation_path}")
    width = int(size_match.group(1))
    height = int(size_match.group(2))

    boxes = []
    pattern = re.compile(
        r"Bounding box for object \d+ .*? : \((\d+), (\d+)\) - \((\d+), (\d+)\)"
    )
    for match in pattern.finditer(text):
        xmin, ymin, xmax, ymax = map(int, match.groups())
        boxes.append((xmin, ymin, xmax, ymax))
    return (width, height), boxes


def boxes_to_yolo_lines(
    image_size: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    cls_id: int = 0,
) -> list[str]:
    """Chuyển list boxes (xmin, ymin, xmax, ymax) sang YOLO format strings."""
    width, height = image_size
    lines = []
    for xmin, ymin, xmax, ymax in boxes:
        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_w = (xmax - xmin) / width
        box_h = (ymax - ymin) / height
        lines.append(
            f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        )
    return lines
