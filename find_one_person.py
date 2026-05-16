import os, glob
d="/home/iec/disk2/TrongTV/atbmht/Naturalistic-Modern-YOLO-Attack/dataset/inria/Train/pos/yolo-labels_modern_gt"
for f in sorted(glob.glob(d+"/*.txt")):
    with open(f) as fp:
        lines = [l for l in fp.readlines() if l.strip()]
        if len(lines) == 1:
            print(os.path.basename(f).replace(".txt", ".png"))
            break
