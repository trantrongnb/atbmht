# Naturalistic Modern YOLO Attack

Thư mục này là bản tách riêng của nhánh tấn công YOLO đời mới, đặt tại:

`/home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack`

Hiện tại repo này đã được gom để có thể tự chạy độc lập trong chính thư mục này:

- `dataset/inria` chứa dữ liệu INRIA
- `adversarialYolo/` chứa `PatchTransformer` và `PatchApplier`
- `GANLatentDiscovery/` chứa code và weights BigGAN
- `ultralytics/` chứa source YOLO dùng cho v8/v9/v10

## Cấu trúc

- `modern_yolo_attack/prepare_inria_labels.py`
  Sinh label YOLO sạch từ annotation gốc của INRIA trong repo hiện tại.
- `modern_yolo_attack/train_biggan_modern_patch.py`
  Train patch cho `yolov5/8/9/10`.
- `modern_yolo_attack/demo_modern_patch.py`
  Test patch trên một ảnh.
- `modern_yolo_attack/batch_eval_modern_patch.py`
  Benchmark patch trên nhiều ảnh INRIA.

## Ghi chú đường dẫn

- Dataset:
  `/home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack/dataset/inria`
- BigGAN:
  `/home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack/GANLatentDiscovery`
- YOLO source:
  `/home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack/ultralytics`
- Output:
  `weights/`, `demo_outputs/`, `batch_outputs/`, `exp/`

## Chuẩn bị

Chạy từ thư mục này:

```bash
cd /home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack
```

Sinh label sạch cho nhánh modern:

```bash
conda run -n signclip_310 python -m modern_yolo_attack.prepare_inria_labels
```

Nếu muốn train bằng `BigGAN latent`, cần có BigGAN weights trong repo này:

```bash
cd /home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack
bash download_biggan.sh
```

## Train nhanh

Chế độ chạy được ngay, không cần BigGAN:

```bash
cd /home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack
conda run -n signclip_310 python -m modern_yolo_attack.train_biggan_modern_patch \
  --generator-mode raw \
  --model yolov8n \
  --epochs 5 \
  --batch-size 4 \
  --limit 32 \
  --max-batches-per-epoch 8
```

Chế độ latent BigGAN:

```bash
cd /home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack
conda run -n signclip_310 python -m modern_yolo_attack.train_biggan_modern_patch \
  --generator-mode biggan \
  --model yolov8n \
  --epochs 5 \
  --batch-size 4 \
  --limit 32 \
  --max-batches-per-epoch 8
```

## Demo

```bash
cd /home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack
conda run -n signclip_310 python -m modern_yolo_attack.demo_modern_patch \
  --model yolov8n \
  --image /home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack/adversarialYolo/data/person.jpg \
  --patch /home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack/exp/yolov8n_biggan_ap/generated/patch-best.png
```

## Batch eval

```bash
cd /home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack
conda run -n signclip_310 python -m modern_yolo_attack.batch_eval_modern_patch \
  --model yolov8n \
  --patch /home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack/exp/yolov8n_biggan_ap/generated/patch-best.png \
  --split Train \
  --limit 50 \
  --batch-size 4
```
