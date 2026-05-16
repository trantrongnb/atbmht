from __future__ import annotations

import argparse
from pathlib import Path

from .common import DATASET_ROOT, DEFAULT_LABEL_DIR_NAME, boxes_to_yolo_lines, parse_inria_annotation


def generate_split(split: str, label_dir_name: str) -> tuple[int, Path]:
    split_root = DATASET_ROOT / split
    annotations_dir = split_root / "annotations"
    output_dir = split_root / "pos" / label_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for annotation_path in sorted(annotations_dir.glob("*.txt")):
        image_size, boxes = parse_inria_annotation(annotation_path)
        lines = boxes_to_yolo_lines(image_size, boxes)
        out_path = output_dir / annotation_path.name
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        count += 1
    return count, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clean YOLO labels from INRIA annotation files.")
    parser.add_argument(
        "--label-dir-name",
        default=DEFAULT_LABEL_DIR_NAME,
        help="Output label directory name created under dataset/inria/<split>/pos/.",
    )
    args = parser.parse_args()

    train_count, train_dir = generate_split("Train", args.label_dir_name)
    test_count, test_dir = generate_split("Test", args.label_dir_name)

    print(f"Generated {train_count} train labels in {train_dir}")
    print(f"Generated {test_count} test labels in {test_dir}")


if __name__ == "__main__":
    main()
