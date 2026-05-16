#!/bin/bash
# =============================================================================
# Download BigGAN weights cho Naturalistic-Modern-YOLO-Attack
# Thư mục đích: Naturalistic-Adversarial-Patch/GANLatentDiscovery/models/pretrained/
#
# BigGAN ch96 (128x128 output, dim_z=120) – dùng trong paper gốc ICCV 2021
# Nguồn: Dropbox (gốc) → HuggingFace (fallback)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
FALLBACK_OLD_REPO_ROOT="$WORKSPACE_ROOT/Naturalistic-Adversarial-Patch"
OLD_REPO_ROOT="$SCRIPT_DIR"
if [ ! -d "$SCRIPT_DIR/GANLatentDiscovery" ]; then
    OLD_REPO_ROOT="$FALLBACK_OLD_REPO_ROOT"
fi

PRETRAINED_DIR="$OLD_REPO_ROOT/GANLatentDiscovery/models/pretrained"
GENERATORS_DIR="$PRETRAINED_DIR/generators/BigGAN"
DEFORMATORS_DIR="$PRETRAINED_DIR/deformators/BigGAN"

echo "=== BigGAN Weight Downloader ==="
echo "Target: $PRETRAINED_DIR"

# Kiểm tra nếu G_ema.pth đã tồn tại
if [ -f "$GENERATORS_DIR/G_ema.pth" ]; then
    echo "[OK] G_ema.pth đã tồn tại tại $GENERATORS_DIR/G_ema.pth"
    echo "     Bỏ qua download. Dùng --force để tải lại."
    SKIP_G=true
else
    SKIP_G=false
fi

# Kiểm tra deformator
if [ -f "$DEFORMATORS_DIR/deformator_0.pt" ] || ls "$DEFORMATORS_DIR"/deformator_*.pt 2>/dev/null | head -1; then
    echo "[OK] Deformator weights đã tồn tại tại $DEFORMATORS_DIR/"
    SKIP_D=true
else
    SKIP_D=false
fi

if [ "$SKIP_G" = true ] && [ "$SKIP_D" = true ]; then
    echo ""
    echo "=== Tất cả weights đã có sẵn. ==="
    exit 0
fi

mkdir -p "$GENERATORS_DIR"
mkdir -p "$DEFORMATORS_DIR"

# ==========================================================================
# Tải BigGAN Generator weights (G_ema.pth)
# BigGAN-deep-256 (ch96, 128px resolution variant) từ pretrained_biggan.tar
# ==========================================================================
if [ "$SKIP_G" = false ]; then
    echo ""
    echo "--- Tải BigGAN Generator weights ---"

    # Thử nguồn Dropbox gốc trước
    DROPBOX_URL="https://www.dropbox.com/s/zte4oein08ajsij/pretrained_biggan.tar?dl=1"
    BIGGAN_TAR="/tmp/pretrained_biggan_$$.tar"

    echo "Thử Dropbox..."
    HTTP_CODE=$(curl -o /dev/null -s -w "%{http_code}" --max-time 15 "$DROPBOX_URL" 2>/dev/null || echo "000")

    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ]; then
        echo "Dropbox có thể truy cập. Đang tải (~500MB)..."
        wget -c "$DROPBOX_URL" -O "$BIGGAN_TAR" --show-progress
        echo "Đang giải nén..."
        cd /tmp
        tar xf "$BIGGAN_TAR"
        # Di chuyển kết quả
        if [ -d "/tmp/pretrained/generators/BigGAN" ]; then
            cp -r /tmp/pretrained/generators/BigGAN/* "$GENERATORS_DIR/"
        fi
        if [ -d "/tmp/pretrained/deformators/BigGAN" ]; then
            cp -r /tmp/pretrained/deformators/BigGAN/* "$DEFORMATORS_DIR/"
        fi
        rm -f "$BIGGAN_TAR"
        rm -rf /tmp/pretrained
    else
        echo "[!] Dropbox không khả dụng (HTTP $HTTP_CODE). Thử HuggingFace..."

        # HuggingFace: BigGAN-deep-256 (PyTorch)
        # pretrained_biggan.tar chứa BigGAN ch96 128-resolution generator
        # Ta tải G_ema.pth từ HuggingFace hub
        HF_BASE="https://huggingface.co/datasets/hysts/biggan-pytorch/resolve/main"

        echo "Tải G_ema.pth từ HuggingFace (BigGAN ch96, resolution=128)..."
        wget -c "${HF_BASE}/G_ema.pth" -O "$GENERATORS_DIR/G_ema.pth" --show-progress || {
            echo ""
            echo "============================================================"
            echo "  KHÔNG TỰ ĐỘNG TẢI ĐƯỢC weights từ cả Dropbox lẫn HuggingFace."
            echo ""
            echo "  Bạn cần tải thủ công G_ema.pth và đặt vào:"
            echo "  $GENERATORS_DIR/G_ema.pth"
            echo ""
            echo "  Các nguồn để tải:"
            echo "  1. Dropbox gốc: https://www.dropbox.com/s/zte4oein08ajsij/pretrained_biggan.tar"
            echo "  2. GitHub: https://github.com/anvoynov/GANLatentDiscovery"
            echo "  3. Liên hệ tác giả paper hoặc dùng torch.hub:"
            echo "     python -c \\"
            echo "       'import torch; G = torch.hub.load(\"huggingface/pytorch-transformers\", ...)'"
            echo "============================================================"
            exit 1
        }
    fi
fi

# ==========================================================================
# Kiểm tra deformator (phải có sau khi giải nén tar)
# Nếu chỉ có G_ema.pth mà không có deformator, training vẫn chạy được
# vì code dùng latent_shift trực tiếp (không qua deformator)
# ==========================================================================
if [ "$SKIP_D" = false ]; then
    echo ""
    echo "--- Kiểm tra Deformator ---"
    if ls "$DEFORMATORS_DIR"/deformator_*.pt 2>/dev/null | head -1 > /dev/null 2>&1; then
        echo "[OK] Deformator weights tìm thấy."
    else
        echo "[INFO] Không tìm thấy deformator_*.pt"
        echo "       Training vẫn chạy được: code dùng G_ema trực tiếp với latent_shift."
        echo "       Nếu bạn cần deformator, tải toàn bộ pretrained_biggan.tar."
    fi
fi

# ==========================================================================
# Xác minh cuối cùng
# ==========================================================================
echo ""
echo "=== Kiểm tra kết quả ==="
if [ -f "$GENERATORS_DIR/G_ema.pth" ]; then
    SIZE=$(du -sh "$GENERATORS_DIR/G_ema.pth" | cut -f1)
    echo "[OK] G_ema.pth ($SIZE) → $GENERATORS_DIR/G_ema.pth"
else
    echo "[FAIL] Không tìm thấy G_ema.pth!"
    exit 1
fi

echo ""
echo "=== Xong! ==="
echo "Bây giờ có thể chạy training với BigGAN:"
echo ""
echo "  cd $SCRIPT_DIR"
echo "  conda run -n signclip_310 python -m modern_yolo_attack.train_biggan_modern_patch \\"
echo "    --generator-mode biggan --model yolov8n --epochs 50 --batch-size 4"
