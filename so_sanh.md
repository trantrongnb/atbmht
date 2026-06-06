# Đánh giá hiệu năng tấn công Adversarial Patch

## 1. Bảng đánh giá độ sụt giảm mAP của 3 mô hình trên 6 patch
Ma trận thể hiện điểm sụt giảm mAP50 (Clean mAP50 - Patched mAP50).

| Mô hình \ Patch | v8n_best | v8n_last | v9n_best | v9n_last | v10n_best | v10n_last |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8n** | **-36.13%** | **-35.44%** | **-14.44%** | **-14.78%** | **-29.40%** | **-29.87%** |
| **yolov9n** | **-29.08%** | **-30.47%** | **-21.89%** | **-22.01%** | **-27.05%** | **-27.64%** |
| **yolov10n** | **-34.44%** | **-35.48%** | **-15.01%** | **-15.86%** | **-33.25%** | **-34.16%** |


## 2. So sánh với phương pháp khác (Patch_tot nhat)

Dưới đây là bảng đánh giá hiệu năng mAP của các mô hình YOLO bị tấn công bởi phương pháp khác (Ghi chú: Các số liệu thể hiện **mAP còn lại**).

| Mô hình | Scale 0.2 (Bản vá lớn hơn - Bảng 9) <br> Tấn công bởi **Patch_tot nhat** | Scale 0.25 (Bản vá rất lớn - Bảng 5) <br> Tấn công bởi **Patch_tot nhat_0.25** |
| :--- | :---: | :---: |
| **YOLOv8n** | **31.27%** | **24.77%** |
| **YOLOv9n** | **41.30%** | **33.13%** |
| **YOLOv10n**| **24.51%** *(Sụt giảm mạnh)* | **14.94%** *(Chạm đáy - Bị "mù")* |

*Nhận xét:* Ở kích thước bản vá $0.25$, hệ thống nhận diện của cả 3 mô hình gần như bị vô hiệu hóa ("mù" hoàn toàn), trong đó **YOLOv10n** thể hiện sự kém chống chịu nhất trước các dạng tấn công dạng bản vá (Adversarial Patch)..


---

---

## 3. Đánh giá và So sánh với phương pháp SOTA (Patch_tot_nhat) ở các Scale 0.2, 0.22 và 0.25

Kết quả đánh giá trên tập INRIA Train, so sánh trực tiếp mAP còn lại (%) của hệ thống giữa 2 phương pháp.

| Mô hình | Clean mAP50 | Our (0.2) | SOTA (0.2) | Our (0.22) | SOTA (0.22) | Our (0.25) | SOTA (0.25) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8n** | 69.74% | 34.60% | **31.27%** | 35.03% | **31.27%** | 41.97% | **23.40%** |
| **yolov9n** | 71.02% | 49.41% | **41.30%** | 50.31% | **41.30%** | 53.55% | **39.97%** |
| **yolov10n** | 69.46% | 35.05% | **24.51%** | 37.58% | **24.51%** | 41.14% | **28.62%** |

*Nhận xét:* Tỷ lệ mAP càng thấp chứng tỏ khả năng tấn công càng mạnh. Các con số in đậm thể hiện phương pháp có khả năng qua mặt (fool) mô hình tốt hơn tại cùng kích thước bản vá.
