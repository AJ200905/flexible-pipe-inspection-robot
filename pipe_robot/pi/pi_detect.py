#!/usr/bin/env python3
"""
pi_detect.py  -  run the YOLO26n NCNN model on the Pi and watch the results
                 on your laptop.

Test on a recorded video first (you can repeat the exact same run):
    python3 pi_detect.py --model best_ncnn_model --src pipe_run1.mp4

Then live camera:
    python3 pi_detect.py --model best_ncnn_model --src auto

Live, undistorted, saving the annotated video:
    python3 pi_detect.py --model best_ncnn_model --src auto \
                         --calib fisheye_calib.npz --save run1.mp4

Only look for cracks and holes:
    python3 pi_detect.py --model best_ncnn_model --src auto --only crack hole

Stop the robot automatically when a crack is seen (ESP32 on USB serial):
    python3 pi_detect.py --model best_ncnn_model --src auto --stop-on crack

Watch it here:  http://jellyfish.local:5000

Every detection is appended to detections.csv, and one picture per second at
most is written into detections/.

It runs on your laptop too - point --model at best.pt there (much slower than
the NCNN folder on the Pi).
"""
import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

from cam_utils import Camera
from stream_server import StreamServer


# --------------------------------------------------------------------- setup
def build_parser():
    p = argparse.ArgumentParser(
        description="YOLO26n crack detection on the pipe inspection robot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="best_ncnn_model",
                   help="NCNN folder (fast on the Pi) or best.pt (laptop)")
    p.add_argument("--src", default="auto",
                   help="auto | usb | csi | a video file name")
    p.add_argument("--imgsz", type=int, default=320,
                   help="keep this the same as the training size")
    p.add_argument("--conf", type=float, default=0.40,
                   help="minimum confidence 0..1 (raise = fewer false alarms)")
    p.add_argument("--only", nargs="*", default=None,
                   help="only these classes, e.g. --only crack hole")
    p.add_argument("--calib", default=None,
                   help="fisheye calibration file from Part 2 (.npz)")
    p.add_argument("--save", default=None,
                   help="save the annotated video (.mp4)")
    p.add_argument("--save-fps", type=float, default=10.0,
                   help="frame rate written into --save")
    p.add_argument("--port", type=int, default=5000,
                   help="web port for the video stream")
    p.add_argument("--stop-on", nargs="*", default=None,
                   help="send STOP to the robot when these classes appear")
    p.add_argument("--robot-port", default=None,
                   help="ESP32 serial port (default: first /dev/ttyUSB*)")
    p.add_argument("--csv", default="detections.csv",
                   help="log file for every detection")
    p.add_argument("--snap-dir", default="detections",
                   help="folder for detection pictures")
    p.add_argument("--no-snap", action="store_true",
                   help="do not save detection pictures")
    p.add_argument("--width", type=int, default=640, help="capture width")
    p.add_argument("--height", type=int, default=480, help="capture height")
    p.add_argument("--loop-video", action="store_true",
                   help="restart a video file at the end (for long tests)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N frames (0 = run until Ctrl+C)")
    p.add_argument("--list-classes", action="store_true",
                   help="print the model's classes and exit")
    return p


def pick_ids(names, wanted, flag):
    """Map class names from the command line to the model's class ids."""
    if not wanted:
        return None
    known = {str(n): i for i, n in names.items()}
    unknown = [w for w in wanted if w not in known]
    if unknown:
        raise SystemExit(
            f"error: --{flag} {' '.join(unknown)}: not a class in this model.\n"
            f"       model classes: {' '.join(sorted(known))}")
    print(f"[model] --{flag} {' '.join(wanted)} -> ids "
          f"{[known[w] for w in wanted]}")
    return [known[w] for w in wanted]


def check_model_path(path):
    if os.path.isdir(path):
        # ultralytics writes model.ncnn.param / model.ncnn.bin inside the folder
        need = ("model.ncnn.param", "model.ncnn.bin")
        missing = [f for f in need if not os.path.exists(os.path.join(path, f))]
        if missing:
            listing = ", ".join(sorted(os.listdir(path))[:6]) or "empty folder"
            raise SystemExit(
                f"error: {path}/ does not look like an NCNN model (missing {missing})\n"
                f"       it contains: {listing}\n"
                "       unzip it with:  unzip best_ncnn_model.zip -d best_ncnn_model")
        return
    if os.path.isfile(path):
        if os.path.basename(path).endswith(".pt"):
            print("[model] warning: a .pt file is much slower than the NCNN "
                  "folder on the Pi - use best_ncnn_model/ if you have it.")
        return
    raise SystemExit(f"error: model not found: {path}")


# ---------------------------------------------------------------------- main
def main():
    args = build_parser().parse_args()
    if not 0.0 < args.conf <= 1.0:
        raise SystemExit("error: --conf must be between 0 and 1")

    check_model_path(args.model)
    model = YOLO(args.model, task="detect")
    names = model.names
    print(f"[model] {args.model} loaded, {len(names)} classes: "
          + ", ".join(f"{i}:{n}" for i, n in names.items()))

    if args.list_classes:
        return

    keep_ids = pick_ids(names, args.only, "only")
    stop_ids = pick_ids(names, args.stop_on, "stop-on")

    print(f"[run] src={args.src}  imgsz={args.imgsz}  conf={args.conf}"
          + (f"  only={args.only}" if args.only else "")
          + (f"  stop-on={args.stop_on}" if args.stop_on else ""))

    robot = None
    if stop_ids:
        # do this before the camera and the web server exist, so a missing ESP32
        # fails fast without leaving anything open
        from robot_serial import Robot
        robot = Robot(args.robot_port)          # raises if the ESP32 is missing
        print("[robot] the robot will be stopped the moment "
              + "/".join(args.stop_on) + " appears")

    cam = Camera(args.src, calib=args.calib, loop_video=args.loop_video,
                 width=args.width, height=args.height)
    server = None
    if args.port > 0:
        server = StreamServer(args.port)
        try:
            server.start()
        except (Exception, SystemExit) as exc:
            # werkzeug calls sys.exit(1) when the port is busy or not allowed,
            # and SystemExit is not an Exception - catch both so that detection
            # keeps running without the web view
            why = ("port unavailable" if isinstance(exc, SystemExit) else exc)
            print(f"[stream] no web view on port {args.port} ({why}).\n"
                  f"          detection continues; another copy may already be "
                  f"using that port, or pass --port 0 to skip the stream.")
            server = None

    os.makedirs(args.snap_dir, exist_ok=True)
    new_log = not os.path.exists(args.csv)
    log_file = open(args.csv, "a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["time", "class", "confidence", "x1", "y1", "x2", "y2", "image"])

    writer = None
    save_failed = False
    if args.save:
        if os.path.splitext(args.save)[1].lower() not in (".mp4", ".avi", ".mkv"):
            args.save += ".mp4"
        print(f"[run] saving annotated video to {args.save} "
              f"at {args.save_fps:g} fps")

    stop_sent = False          # True from the moment we stop until the defect goes
    last_snap = 0.0
    frame_no = 0
    n_logged = 0
    fps = 0.0
    t_prev = time.time()
    t_report = t_prev
    running = True

    def stop_now(signum, frame):                # Ctrl+C and 'kill' both exit cleanly
        nonlocal running
        running = False
        print(f"\n[run] signal {signum} - finishing up ...")

    # Ctrl+C and 'kill' both finish the run cleanly
    try:
        signal.signal(signal.SIGINT, stop_now)
        signal.signal(signal.SIGTERM, stop_now)
    except ValueError:
        pass                    # not the main thread (tests, notebooks)

    print(f"[run] running. Ctrl+C to stop. Log: {args.csv}"
          + (f"   Watch: http://<this-pi>:{args.port}/" if server else ""))
    try:
        while running:
            frame = cam.read()
            if frame is None:
                if cam.mode == "file":
                    print("[run] end of the video file.")
                else:
                    print("[run] no frame from the camera - stopping.")
                break
            frame_no += 1

            r = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                              classes=keep_ids, verbose=False)[0]
            annotated = r.plot()

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            found = [names[int(b.cls)] for b in r.boxes]
            counts = {}
            for n in found:
                counts[n] = counts.get(n, 0) + 1
            summary = ", ".join(f"{n}x{c}" if c > 1 else n for n, c in counts.items())

            cv2.putText(annotated, f"{fps:.1f} FPS", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

            # ---- log every detection, save at most one picture per second
            if found:
                img_name = ""
                if not args.no_snap and now - last_snap > 1.0:
                    img_name = os.path.join(args.snap_dir, f"{stamp}.jpg")
                    cv2.imwrite(img_name, annotated)
                    last_snap = now
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    log.writerow([stamp, names[int(b.cls)], f"{float(b.conf):.2f}",
                                  x1, y1, x2, y2, img_name])
                    n_logged += 1
                log_file.flush()

            # ---- stop the robot once per defect event, re-arm when it is gone.
            # This must be checked every frame, also when nothing is detected,
            # otherwise the robot stays latched in STOP for the rest of the run.
            if robot is not None:
                hit = any(int(b.cls) in stop_ids for b in r.boxes)
                if hit and not stop_sent:
                    robot.stop()
                    stop_sent = True
                    print(f"[robot] STOP sent - {summary or 'defect'}")
                elif not hit and stop_sent:
                    stop_sent = False
                    print("[robot] defect gone - re-armed. (No RESET is sent; "
                          "the driver stays stopped until you reset it by hand.)")

            if server is not None:
                server.update(annotated, fps=fps)
                server.info = f"{fps:.1f} FPS | " + (summary if found else "no defects")

            if args.save and not save_failed:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(args.save,
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.save_fps, (w, h))
                    if not writer.isOpened():
                        print(f"[run] cannot write {args.save} - stopping the recording")
                        writer = None
                        save_failed = True      # do not retry every single frame
                if writer is not None:
                    writer.write(annotated)

            # ---- a line every 10 s is handy over SSH
            if now - t_report > 10.0:
                t_report = now
                print(f"[run] frame {frame_no}  {fps:.1f} FPS  "
                      + (summary if found else "no defects"))

            if args.max_frames and frame_no >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n[run] Ctrl+C - stopping.")
    finally:
        if writer is not None:
            writer.release()
        log_file.close()
        cam.release()
        if server is not None:
            server.close()
        if robot is not None:
            robot.close()
        print(f"[run] done. {frame_no} frames, {n_logged} detections logged to "
              f"{args.csv} in this run")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:                # camera / ESP32 problems
        print(f"\nerror: {exc}", file=sys.stderr)
        sys.exit(2)