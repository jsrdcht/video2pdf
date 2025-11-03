import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def find_whiteboard_bbox(frame_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hch, sch, vch = cv2.split(hsv)

    # White-ish: low saturation, high value
    mask = (sch < 40) & (vch > 210)
    mask = mask.astype(np.uint8) * 255

    # Morphology to reduce noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w, h

    areas = [cv2.contourArea(c) for c in contours]
    idx = int(np.argmax(areas))
    x, y, cw, ch = cv2.boundingRect(contours[idx])

    # Slightly expand bbox within image bounds
    pad_x = max(0, int(round(0.01 * w)))
    pad_y = max(0, int(round(0.01 * h)))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + cw + pad_x)
    y1 = min(h, y + ch + pad_y)
    return x0, y0, x1 - x0, y1 - y0


def main() -> int:
    video = Path('E:/video2pdf/input.mp4')
    if not video.exists():
        print('ERR: E:/video2pdf/input.mp4 not found', file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print('ERR: cannot open video', file=sys.stderr)
        return 3
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print('ERR: cannot read first frame', file=sys.stderr)
        return 4

    x, y, w, h = find_whiteboard_bbox(frame)
    print(f"CROP {x},{y},{w},{h}")

    # Save a preview image with rectangle overlay
    preview = frame.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 4)
    out_path = Path('E:/video2pdf/crop_preview.png')
    cv2.imwrite(str(out_path), preview)
    print(f"Preview saved to {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


