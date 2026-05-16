import sys
import glob
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

PROJECT_ROOT = Path("/home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack")
for _candidate in (PROJECT_ROOT, PROJECT_ROOT / "adversarialYolo", PROJECT_ROOT / "ultralytics"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from adversarialYolo.load_data import PatchApplier, PatchTransformer
from modern_yolo_attack.common import load_modern_detector, to_square_patch_tensor

def draw_yolo_result(img_tensor, result, title):
    img_np = (img_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.putText(img_bgr, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(img_bgr, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    found_person = False
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            if int(box.cls.cpu().item()) == 0:
                conf = float(box.conf.cpu().item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"Person: {conf:.2f}"
                cv2.putText(img_bgr, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(img_bgr, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                found_person = True
    if not found_person:
        cv2.putText(img_bgr, "FOOLED!", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    return img_bgr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, _ = load_modern_detector("yolov8n", device, PROJECT_ROOT / "weights")
    detector.model.eval()
    
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
    
    best_patch_path = PROJECT_ROOT / "exp" / "yolov8n_biggan_ap" / "generated" / "patch-best.png"
    best_patch = to_square_patch_tensor(best_patch_path, device)
    
    label_dir = PROJECT_ROOT / "dataset" / "inria" / "Train" / "pos" / "yolo-labels_modern_gt"
    img_dir = PROJECT_ROOT / "dataset" / "inria" / "Train" / "pos"
    
    best_img_path = None
    best_score_diff = -1
    best_lab_tensor = None
    best_img_tensor = None
    
    to_tensor = T.ToTensor()
    
    count = 0
    # Quét qua ảnh để tìm bức ảnh nào có 1 người mà patch lừa được đỉnh nhất (score rớt từ > 0.8 xuống 0)
    for txt_path in sorted(glob.glob(str(label_dir / "*.txt"))):
        with open(txt_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and l.startswith("0 ")]
        
        if len(lines) == 1:
            img_path = img_dir / Path(txt_path).name.replace(".txt", ".png")
            if not img_path.exists(): continue
            
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = to_tensor(img_rgb).unsqueeze(0).to(device)
            img_tensor = F.interpolate(img_tensor, size=(416, 416), mode="bilinear", align_corners=False)
            
            # Clean detection
            res_clean = detector(img_tensor, verbose=False, conf=0.25)[0]
            clean_conf = 0
            if res_clean.boxes is not None and len(res_clean.boxes) > 0:
                for box in res_clean.boxes:
                    if int(box.cls.cpu().item()) == 0:
                        clean_conf = max(clean_conf, float(box.conf.cpu().item()))
            
            if clean_conf < 0.8: continue # Chỉ tìm ảnh nào ban đầu nhận diện cực kỳ nét
            
            parts = lines[0].split()
            lab_tensor = torch.tensor([[[0, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]]], dtype=torch.float32).to(device)
            
            # Patched detection
            adv_batch_p, _, _ = patch_transformer(
                adv_patch=best_patch[0], lab_batch=lab_tensor, img_size=416,
                patch_mask=[], by_rectangle=True, do_rotate=False, rand_loc=False,
                with_black_trans=False, scale_rate=0.2, enable_blurred=False
            )
            img_best = patch_applier(img_tensor, adv_batch_p)
            res_best = detector(img_best, verbose=False, conf=0.25)[0]
            patched_conf = 0
            if res_best.boxes is not None and len(res_best.boxes) > 0:
                for box in res_best.boxes:
                    if int(box.cls.cpu().item()) == 0:
                        patched_conf = max(patched_conf, float(box.conf.cpu().item()))
            
            diff = clean_conf - patched_conf
            if diff > best_score_diff:
                best_score_diff = diff
                best_img_path = img_path
                best_lab_tensor = lab_tensor
                best_img_tensor = img_tensor
                
                # Nếu lừa thành công 100% (rớt xuống 0) thì chốt luôn ảnh này
                if patched_conf == 0:
                    break
            
            count += 1
            if count > 200: break
            
    print(f"Bức ảnh tốt nhất: {best_img_path.name} (Tụt {best_score_diff:.3f} điểm)")
    
    # 1. Clean
    res_clean = detector(best_img_tensor, verbose=False, conf=0.25)[0]
    img_clean_bgr = draw_yolo_result(best_img_tensor, res_clean, "1. Clean (No Patch)")
    
    # 2. Black
    black_patch = torch.zeros((1, 3, 128, 128), device=device)
    adv_batch_b, _, _ = patch_transformer(
        adv_patch=black_patch[0], lab_batch=best_lab_tensor, img_size=416,
        patch_mask=[], by_rectangle=True, do_rotate=False, rand_loc=False,
        with_black_trans=False, scale_rate=0.2, enable_blurred=False
    )
    img_black = patch_applier(best_img_tensor, adv_batch_b)
    res_black = detector(img_black, verbose=False, conf=0.25)[0]
    img_black_bgr = draw_yolo_result(img_black, res_black, "2. Black Patch")
    
    # 3. Best Patch
    adv_batch_p, _, _ = patch_transformer(
        adv_patch=best_patch[0], lab_batch=best_lab_tensor, img_size=416,
        patch_mask=[], by_rectangle=True, do_rotate=False, rand_loc=False,
        with_black_trans=False, scale_rate=0.2, enable_blurred=False
    )
    img_best_p = patch_applier(best_img_tensor, adv_batch_p)
    res_best_p = detector(img_best_p, verbose=False, conf=0.25)[0]
    img_best_bgr = draw_yolo_result(img_best_p, res_best_p, "3. Adversarial Patch")
    
    out_dir = Path("/home/iec/.gemini/antigravity/brain/0c789177-8f25-445b-930e-43eb5267c01f")
    cv2.imwrite(str(out_dir / "train_yolov8_clean.png"), img_clean_bgr)
    cv2.imwrite(str(out_dir / "train_yolov8_black.png"), img_black_bgr)
    cv2.imwrite(str(out_dir / "train_yolov8_best.png"), img_best_bgr)

if __name__ == "__main__":
    main()
