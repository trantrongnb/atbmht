import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

csv_path = Path("exp/yolov8n_biggan_ap/history.csv")
history_rows = []
with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        converted = {}
        for k, v in row.items():
            try:
                converted[k] = float(v) if v else None
            except:
                converted[k] = v
        history_rows.append(converted)

epochs = [row["epoch"] for row in history_rows]
total_loss = [row["mean_loss_total"] for row in history_rows]
attack_obj = [row["mean_attack_obj"] for row in history_rows]
detector_loss = [row["mean_detector_train_loss"] for row in history_rows]
tv_loss = [row["mean_loss_tv"] for row in history_rows]
nps_loss = [row["mean_loss_nps"] for row in history_rows]
lrs = [row["lr"] for row in history_rows]

# Trích xuất riêng các epoch có eval AP để vẽ liền mạch
ap_epochs = []
ap50_values = []
ap50_95_values = []
for row in history_rows:
    if row.get("eval_mean_ap50") is not None:
        ap_epochs.append(row["epoch"])
        ap50_values.append(row["eval_mean_ap50"])
        ap50_95_values.append(row["eval_mean_ap50_95"])

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.ravel()

axes[0].plot(epochs, total_loss, label="total")
axes[0].plot(epochs, detector_loss, label="detector_train")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(epochs, attack_obj, color="tab:red")
axes[1].set_title("Attack Objective (minimize)")
axes[1].set_xlabel("Epoch")
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs, tv_loss, label="tv")
axes[2].plot(epochs, nps_loss, label="nps")
axes[2].set_title("Regularization")
axes[2].set_xlabel("Epoch")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

# Vẽ AP bằng marker (o-, s-) để nối các đường lại với nhau bỏ qua epoch trống
if ap_epochs:
    axes[3].plot(ap_epochs, ap50_values, 'o-', label="AP50", linewidth=2)
    axes[3].plot(ap_epochs, ap50_95_values, 's-', label="AP50-95", linewidth=2)
    axes[3].set_title("Evaluation AP")
    axes[3].set_xlabel("Epoch")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

axes[4].plot(epochs, lrs, color="tab:green", linewidth=2)
axes[4].set_title("Learning Rate")
axes[4].set_xlabel("Epoch")
axes[4].grid(True, alpha=0.3)

axes[5].axis("off")

fig.tight_layout()
fig.savefig("exp/yolov8n_biggan_ap/history.png", dpi=160)
plt.close(fig)
print("Đã vẽ lại biểu đồ thành công!")
