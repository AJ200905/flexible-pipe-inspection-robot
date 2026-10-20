#!/usr/bin/env python3
"""
make_test_video.py - render a fake pipe-tunnel video for testing.

Useful before any real footage exists: it gives pi_detect.py a video file with
moving pipe walls, a crack, a joint offset and some debris, so the whole chain
(camera -> model -> boxes -> stream -> csv) can be checked on a known input.

    python3 make_test_video.py test_tunnel.mp4 --seconds 10

It is only a plumbing test - the model will not detect anything meaningful in
a synthetic image, but it proves the code path works.
"""
import argparse
import math
import os

import cv2
import numpy as np


def pipe_texture(w=960, h=720, seed=7):
    """One 'wall' image: pipe rings, grain, a bright light streak."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    theta = np.arctan2(yy - h / 2, xx - w / 2)

    shade = 60 + 45 * np.cos(theta * 3) + 30 * np.cos(r / 26.0)
    for c in range(3):                                  # slightly warm, damp concrete
        img[..., c] = np.clip(shade * (1.0 + 0.05 * c), 0, 255)

    # dark ring every ~110 px = the pipe joints
    for radius in range(90, int(r.max()), 110):
        ring = (r > radius) & (r < radius + 12)
        img[ring] = img[ring] * 0.45

    # lamp reflection down the middle
    lamp = np.exp(-((xx - w / 2) ** 2) / (2 * 55.0 ** 2))
    img = np.clip(img.astype(np.float32) + 70 * lamp[..., None], 0, 255)
    img += rng.normal(0, 7, img.shape).astype(np.float32)   # grain
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_crack(img, cx, cy, length=120, angle=-0.5, width=3, seed=1):
    """A jagged dark crack, drawn straight onto the wall."""
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(14):
        f = i / 13.0
        wobble = rng.normal(0, 7)
        x = cx + math.cos(angle) * length * (f - 0.5) - math.sin(angle) * wobble
        y = cy + math.sin(angle) * length * (f - 0.5) + math.cos(angle) * wobble
        pts.append((int(x), int(y)))
    cv2.polylines(img, [np.array(pts, np.int32)], False, (18, 16, 14), width, cv2.LINE_AA)
    cv2.polylines(img, [np.array(pts, np.int32)], False, (30, 27, 24), 1, cv2.LINE_AA)
    return pts


def draw_debris(img, cx, cy, seed=2):
    rng = np.random.default_rng(seed)
    for _ in range(9):
        x = int(cx + rng.normal(0, 34))
        y = int(cy + rng.normal(0, 22))
        size = int(rng.integers(6, 17))
        cv2.ellipse(img, (x, y), (size, size // 2), float(rng.uniform(0, 180)),
                    0, 360, (int(rng.integers(30, 70)),) * 3, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="test_tunnel.mp4")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    src = pipe_texture()
    src = cv2.GaussianBlur(src, (3, 3), 0)

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (args.width, args.height))
    if not writer.isOpened():
        raise SystemExit(f"cannot write {args.out} - is OpenCV built with ffmpeg?")

    total = int(args.fps * args.seconds)
    sh, sw = src.shape[:2]
    for i in range(total):
        t = i / max(total - 1, 1)
        # camera moving forward: crop a shrinking window, then blow it back up
        zoom = 1.0 + 0.9 * t
        cw, ch = int(sw / zoom), int(sh / zoom)
        x0, y0 = (sw - cw) // 2, (sh - ch) // 2
        frame = cv2.resize(src[y0:y0 + ch, x0:x0 + cw], (args.width, args.height))

        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        # lamp flicker + vignette, so the test video is not brighter than life
        gain = 1.0 + 0.06 * math.sin(i / 5.0)
        frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        h, w = frame.shape[:2]
        vig = np.hypot(np.mgrid[0:h, 0:w][1] - w / 2, np.mgrid[0:h, 0:w][0] - h / 2)
        falloff = 1 - 0.35 * (vig / (w / 2)) ** 2
        frame = np.clip(frame.astype(np.float32) * falloff[..., None],
                        0, 255).astype(np.uint8)
        cv2.putText(frame, "synthetic test tunnel", (10, args.height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        writer.write(frame)
    writer.release()

    # a second clip where the defect is really there, for a fair "before/after"
    wall = pipe_texture(seed=7)
    draw_crack(wall, 470, 330, length=170, seed=11)
    draw_crack(wall, 500, 470, length=120, angle=0.4, seed=12)
    draw_debris(wall, 300, 250)
    stem, ext = os.path.splitext(args.out)
    defects_path = (stem + "_defects" + (ext or ".mp4"))
    writer = cv2.VideoWriter(defects_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise SystemExit(f"cannot write {defects_path}")
    for i in range(total):
        t = i / max(total - 1, 1)
        zoom = 1.0 + 0.9 * t
        cw, ch = int(sw / zoom), int(sh / zoom)
        x0, y0 = (sw - cw) // 2, (sh - ch) // 2
        frame = cv2.resize(wall[y0:y0 + ch, x0:x0 + cw], (args.width, args.height))
        frame = np.clip(frame.astype(np.float32) * (1.0 + 0.05 * math.sin(i / 4.0)),
                        0, 255).astype(np.uint8)
        cv2.putText(frame, "synthetic + crack/debris", (10, args.height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        writer.write(frame)
    writer.release()

    print(f"wrote {args.out} and {defects_path} "
          f"({total} frames each at {args.fps} fps)")


if __name__ == "__main__":
    main()