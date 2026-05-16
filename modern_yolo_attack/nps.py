"""
Non-Printability Score (NPS) Loss

Tính NPS để patch sinh ra có thể in được trong thực tế.
Dựa trên: "Fooling automated surveillance cameras: adversarial patches to attack
person detection" (Thys et al., CVPR Workshop 2019)

NPS đo khoảng cách giữa màu của patch với tập màu có thể in (printable colors).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


# Bộ màu in được mặc định (30 màu cơ bản từ printer color gamut)
# Được trích từ adversarialYolo/non_printability/
DEFAULT_PRINTABLE_COLORS = [
    (0.0, 0.0, 0.0),        # black
    (1.0, 1.0, 1.0),        # white
    (1.0, 0.0, 0.0),        # red
    (0.0, 1.0, 0.0),        # green
    (0.0, 0.0, 1.0),        # blue
    (1.0, 1.0, 0.0),        # yellow
    (0.0, 1.0, 1.0),        # cyan
    (1.0, 0.0, 1.0),        # magenta
    (0.502, 0.000, 0.000),  # dark red
    (0.000, 0.502, 0.000),  # dark green
    (0.000, 0.000, 0.502),  # dark blue
    (0.502, 0.502, 0.000),  # olive
    (0.000, 0.502, 0.502),  # teal
    (0.502, 0.000, 0.502),  # purple
    (1.000, 0.502, 0.000),  # orange
    (1.000, 0.000, 0.502),  # rose
    (0.502, 1.000, 0.000),  # chartreuse
    (0.000, 1.000, 0.502),  # spring green
    (0.000, 0.502, 1.000),  # azure
    (0.502, 0.000, 1.000),  # violet
    (0.800, 0.800, 0.800),  # light gray
    (0.600, 0.600, 0.600),  # gray
    (0.400, 0.400, 0.400),  # dark gray
    (0.200, 0.200, 0.200),  # darker gray
    (0.941, 0.502, 0.502),  # light red/pink
    (0.502, 0.941, 0.502),  # light green
    (0.502, 0.502, 0.941),  # light blue
    (0.941, 0.941, 0.502),  # light yellow
    (0.502, 0.941, 0.941),  # light cyan
    (0.941, 0.502, 0.941),  # light magenta
]


def load_printable_colors(nps_file: Path | None = None) -> list[tuple[float, float, float]]:
    """
    Đọc bộ màu in được từ file text (mỗi dòng: R,G,B trong [0,255]).
    Nếu không có file, dùng DEFAULT_PRINTABLE_COLORS.
    """
    if nps_file is None or not nps_file.exists():
        return DEFAULT_PRINTABLE_COLORS

    colors = []
    for line in nps_file.read_text().strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) == 3:
            r, g, b = [float(x.strip()) / 255.0 for x in parts]
            colors.append((r, g, b))
    return colors if colors else DEFAULT_PRINTABLE_COLORS


class NPSCalculator(nn.Module):
    """
    Non-Printability Score Calculator.

    NPS(patch) = Σ_{p in patch} min_{c in C} ||p - c||²
    trong đó C là tập màu có thể in.
    """

    def __init__(
        self,
        printable_colors: list[tuple[float, float, float]] | None = None,
        nps_file: Path | None = None,
    ) -> None:
        super().__init__()

        if printable_colors is None:
            printable_colors = load_printable_colors(nps_file)

        # printability_array: (num_colors, 3)
        colors_tensor = torch.tensor(printable_colors, dtype=torch.float32)
        self.register_buffer("printability_array", colors_tensor)

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: (3, H, W) – pixel values trong [0, 1]

        Returns:
            Scalar NPS loss.
        """
        # adv_patch: (3, H, W) → (H*W, 3)
        patch_flat = adv_patch.permute(1, 2, 0).reshape(-1, 3)  # (N, 3)

        # printability_array: (C, 3)
        # Khoảng cách bình phương: (N, C)
        diff = patch_flat.unsqueeze(1) - self.printability_array.unsqueeze(0)  # (N, C, 3)
        dist_sq = (diff ** 2).sum(dim=2)  # (N, C)

        # Min khoảng cách tới màu gần nhất cho mỗi pixel
        min_dist, _ = dist_sq.min(dim=1)  # (N,)

        return min_dist.mean()
