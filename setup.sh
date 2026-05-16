#!/bin/bash
# =============================================================================
# setup.sh – Cài đặt và chuẩn bị môi trường Naturalistic-Modern-YOLO-Attack
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
FALLBACK_OLD_REPO_ROOT="$WORKSPACE_ROOT/Naturalistic-Adversarial-Patch"
FALLBACK_MODERN_SOURCE_REPO="$WORKSPACE_ROOT/NaturalisticAdversarialPatches"
OLD_REPO_ROOT="$SCRIPT_DIR"
MODERN_SOURCE_REPO="$SCRIPT_DIR"
if [ ! -d "$SCRIPT_DIR/adversarialYolo" ] || [ ! -d "$SCRIPT_DIR/GANLatentDiscovery" ]; then
    OLD_REPO_ROOT="$FALLBACK_OLD_REPO_ROOT"
fi
if [ ! -d "$SCRIPT_DIR/ultralytics" ]; then
    MODERN_SOURCE_REPO="$FALLBACK_MODERN_SOURCE_REPO"
fi
CONDA_ENV="${CONDA_ENV:-signclip_310}"
AUTO_DOWNLOAD_BIGGAN="${AUTO_DOWNLOAD_BIGGAN:-0}"

echo "============================================================"
echo " Naturalistic-Modern-YOLO-Attack – Setup"
echo "============================================================"
echo " Project root : $SCRIPT_DIR"
echo " Legacy root  : $OLD_REPO_ROOT"
echo " YOLO root    : $MODERN_SOURCE_REPO"
echo " Conda env    : $CONDA_ENV"
echo "============================================================"

# ------------------------------------------------------------
# Bước 1: Kiểm tra các thư mục phụ thuộc
# ------------------------------------------------------------
echo ""
echo "[1/5] Kiểm tra thư mục phụ thuộc..."

check_dir() {
    if [ -d "$1" ]; then
        echo "  [OK] $1"
    else
        echo "  [FAIL] Không tìm thấy: $1"
        exit 1
    fi
}

check_dir "$OLD_REPO_ROOT"
check_dir "$OLD_REPO_ROOT/adversarialYolo"
check_dir "$OLD_REPO_ROOT/GANLatentDiscovery"
check_dir "$OLD_REPO_ROOT/dataset/inria"
check_dir "$MODERN_SOURCE_REPO"
check_dir "$MODERN_SOURCE_REPO/ultralytics"

# ------------------------------------------------------------
# Bước 2: Kiểm tra dataset INRIA
# ------------------------------------------------------------
echo ""
echo "[2/5] Kiểm tra dataset INRIA..."

INRIA_TRAIN="$OLD_REPO_ROOT/dataset/inria/Train/pos"
INRIA_TEST="$OLD_REPO_ROOT/dataset/inria/Test/pos"

if [ -d "$INRIA_TRAIN" ] && [ "$(ls -1 "$INRIA_TRAIN"/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
    N_TRAIN=$(ls -1 "$INRIA_TRAIN"/*.png 2>/dev/null | wc -l)
    echo "  [OK] Train: $N_TRAIN ảnh tại $INRIA_TRAIN"
else
    echo "  [FAIL] Không có ảnh train tại $INRIA_TRAIN"
    echo "         Chạy: cd $OLD_REPO_ROOT && bash download_inria.sh"
    exit 1
fi

if [ -d "$INRIA_TEST" ] && [ "$(ls -1 "$INRIA_TEST"/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
    N_TEST=$(ls -1 "$INRIA_TEST"/*.png 2>/dev/null | wc -l)
    echo "  [OK] Test: $N_TEST ảnh tại $INRIA_TEST"
fi

# ------------------------------------------------------------
# Bước 3: Sinh YOLO labels (nếu chưa có)
# ------------------------------------------------------------
echo ""
echo "[3/5] Kiểm tra / sinh YOLO labels..."

LABEL_DIR="$INRIA_TRAIN/yolo-labels_modern_gt"
if [ -d "$LABEL_DIR" ] && [ "$(ls -1 "$LABEL_DIR"/*.txt 2>/dev/null | wc -l)" -gt 0 ]; then
    N_LABELS=$(ls -1 "$LABEL_DIR"/*.txt 2>/dev/null | wc -l)
    echo "  [OK] $N_LABELS label files tại $LABEL_DIR"
else
    echo "  [!] Chưa có labels. Đang sinh..."
    cd "$SCRIPT_DIR"
    conda run -n "$CONDA_ENV" python -m modern_yolo_attack.prepare_inria_labels
    echo "  [OK] Labels đã được sinh."
fi

# ------------------------------------------------------------
# Bước 4: Kiểm tra BigGAN weights
# ------------------------------------------------------------
echo ""
echo "[4/5] Kiểm tra BigGAN weights..."

BIGGAN_WEIGHTS="$OLD_REPO_ROOT/GANLatentDiscovery/models/pretrained/generators/BigGAN/G_ema.pth"

if [ -f "$BIGGAN_WEIGHTS" ]; then
    SIZE=$(du -sh "$BIGGAN_WEIGHTS" | cut -f1)
    echo "  [OK] G_ema.pth ($SIZE) tại $BIGGAN_WEIGHTS"
else
    echo "  [WARN] Chưa có G_ema.pth tại $BIGGAN_WEIGHTS"
    if [ "$AUTO_DOWNLOAD_BIGGAN" = "1" ]; then
        echo "  [!] Đang thử tải BigGAN weights..."
        if bash "$SCRIPT_DIR/download_biggan.sh"; then
            echo "  [OK] BigGAN weights đã sẵn sàng."
        else
            echo "  [WARN] Không tải được BigGAN weights. Raw mode vẫn chạy được."
        fi
    else
        echo "         Raw mode vẫn chạy được ngay."
        echo "         Muốn train BigGAN, chạy:"
        echo "           bash $SCRIPT_DIR/download_biggan.sh"
        echo "         Hoặc:"
        echo "           AUTO_DOWNLOAD_BIGGAN=1 bash $SCRIPT_DIR/setup.sh"
    fi
fi

# ------------------------------------------------------------
# Bước 5: Tạo thư mục output
# ------------------------------------------------------------
echo ""
echo "[5/5] Tạo thư mục output..."

mkdir -p "$SCRIPT_DIR/weights"
mkdir -p "$SCRIPT_DIR/dataset"
mkdir -p "$SCRIPT_DIR/exp"
mkdir -p "$SCRIPT_DIR/demo_outputs"
mkdir -p "$SCRIPT_DIR/batch_outputs"
mkdir -p "$SCRIPT_DIR/runs/tensorboard"
echo "  [OK] weights/, dataset/, exp/, demo_outputs/, batch_outputs/, runs/tensorboard/"

# ------------------------------------------------------------
# Xong
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo " Setup hoàn tất!"
echo ""
echo " Quick start (không cần BigGAN):"
echo "   cd $SCRIPT_DIR"
echo "   conda run -n $CONDA_ENV python -m modern_yolo_attack.train_biggan_modern_patch \\"
echo "     --generator-mode raw --model yolov8n --epochs 5 --batch-size 4 --limit 64"
echo ""
echo " Train với BigGAN:"
echo "   conda run -n $CONDA_ENV python -m modern_yolo_attack.train_biggan_modern_patch \\"
echo "     --generator-mode biggan --model yolov8n --epochs 100 --batch-size 4"
echo ""
echo " Train ensemble (v8n + v9t + v10n):"
echo "   conda run -n $CONDA_ENV python -m modern_yolo_attack.train_biggan_modern_patch \\"
echo "     --generator-mode biggan --ensemble-models yolov8n yolov9t yolov10n --epochs 100"
echo ""
echo " Nếu thiếu tensorboard, script train vẫn chạy nhưng sẽ tắt logging TensorBoard."
echo "============================================================"
