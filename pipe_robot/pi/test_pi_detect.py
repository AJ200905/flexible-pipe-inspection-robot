#!/usr/bin/env python3
"""
test_pi_detect.py - exercise pi_detect.py end to end without a trained model.

A stub model stands in for the YOLO one, so this runs anywhere, even on a Pi
where PyTorch crashes.  It checks the parts that are ours: the argument
handling, the CSV log, the one-picture-per-second rule, the FPS counter, the web
stream, the --only filter and the ESP32 stop/re-arm logic.

    python3 test_pi_detect.py
"""
import csv
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import types
import urllib.error
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pi_detect  # noqa: E402

PASSED = []
FAILED = []


def check(condition, message):
    (PASSED if condition else FAILED).append(message)
    print(f"   {'PASS' if condition else 'FAIL'} {message}", flush=True)


# ------------------------------------------------------------------ stubs
class FakeBox:
    def __init__(self, cls, conf, xyxy):
        self.cls = cls
        self.conf = conf
        self.xyxy = np.array([xyxy], dtype=np.float32)


class FakeResult:
    def __init__(self, frame, boxes):
        self.boxes = boxes

    def plot(self):
        img = self.frame.copy()
        for box in self.boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"class{int(box.cls)}", (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img


class FakeModel:
    """Returns a crack on some frames and nothing on others, on purpose."""

    names = {0: "crack", 1: "hole", 2: "debris"}
    stop_after = 0            # when > 0, raise KeyboardInterrupt after N frames
    counter = 0
    seen_imgsz = None
    seen_conf = None

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, frame, imgsz=320, conf=0.4, classes=None, verbose=False):
        type(self).counter += 1
        FakeModel.seen_imgsz = imgsz
        FakeModel.seen_conf = conf
        n = type(self).counter
        if type(self).stop_after and n >= type(self).stop_after:
            # what Ctrl+C looks like from inside the loop
            raise KeyboardInterrupt
        boxes = []
        if n % 4 in (0, 1):                 # defect on, then off, then on again
            boxes.append(FakeBox(0, 0.83, [10, 10, 90, 80]))
        if n % 4 == 1:
            boxes.append(FakeBox(1, 0.61, [40, 40, 120, 110]))
        if classes is not None:
            boxes = [b for b in boxes if int(b.cls) in classes]
        result = FakeResult(frame, boxes)
        result.frame = frame
        return [result]


class FakeRobot:
    """Records the commands pi_detect sends."""
    events = []

    def __init__(self, port=None, *args, **kwargs):
        FakeRobot.events.append("connect")

    def stop(self):
        FakeRobot.events.append("STOP")
        return "OK"

    def close(self):
        FakeRobot.events.append("close")


MODEL = ["stub"]        # replaced by make_model_folder() in main()


def run(args, expect_rc=0):
    """Run pi_detect.main() with the stub model installed."""
    FakeModel.counter = 0
    FakeModel.stop_after = 0
    pi_detect.YOLO = lambda *a, **k: FakeModel()
    args = [MODEL[0] if a == "stub" else a for a in args]
    old_argv = sys.argv
    sys.argv = ["pi_detect.py"] + args
    try:
        pi_detect.main()
        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.argv = old_argv


def free_port():
    """Ask the OS for a port nobody is using, then let it go."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def make_video(path, frames=60, w=320, h=240, fps=15):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(frames):
        writer.write(np.full((h, w, 3), 40 + i % 60, np.uint8))
    writer.release()


def make_model_folder(root):
    """A folder that passes the real check_model_path, with no weights in it.

    Everything else is faked, but this keeps the model-loading checks real.
    """
    folder = os.path.join(root, "fake_ncnn_model")
    os.makedirs(folder, exist_ok=True)
    for name in ("model.ncnn.param", "model.ncnn.bin"):
        with open(os.path.join(folder, name), "wb") as fh:
            fh.write(b"")
    with open(os.path.join(folder, "metadata.yaml"), "w") as fh:
        fh.write("names: {0: crack, 1: hole, 2: debris}\n")
    return folder


# ------------------------------------------------------------------- tests
def test_model_path_check(tmp):
    """check_model_path is what protects the user from a half-unzipped model."""
    print("\n== model folder checks ==")
    real_check = pi_detect.check_model_path

    folder = os.path.join(tmp, "good_ncnn")
    os.makedirs(folder)
    for name in ("model.ncnn.param", "model.ncnn.bin"):
        open(os.path.join(folder, name), "wb").close()
    try:
        real_check(folder)
        ok = True
    except SystemExit:
        ok = False
    check(ok, "a folder with model.ncnn.param/.bin is accepted")

    empty = os.path.join(tmp, "empty_ncnn")
    os.makedirs(empty)
    try:
        real_check(empty)
        ok = False
    except SystemExit:
        ok = True
    check(ok, "an empty folder is rejected with a message")

    try:
        real_check(os.path.join(tmp, "not_here"))
        ok = False
    except SystemExit:
        ok = True
    check(ok, "a missing model path is rejected")

    weights = os.path.join(tmp, "best.pt")
    open(weights, "wb").close()
    try:
        real_check(weights)
        ok = True
    except SystemExit:
        ok = False
    check(ok, "a .pt file is accepted (with a warning)")


def test_csv_and_snapshots(tmp, video):
    print("\n== csv, snapshots and options ==")
    csv_path = os.path.join(tmp, "detections.csv")
    snap_dir = os.path.join(tmp, "detections")
    rc = run(["--model", "stub", "--src", video, "--port", "0", "--conf", "0.4",
              "--imgsz", "320", "--csv", csv_path, "--snap-dir", snap_dir,
              "--max-frames", "40"])
    check(rc == 0, f"run finished cleanly (rc={rc})")
    check(os.path.exists(csv_path), "detections.csv was created")

    rows = read_csv(csv_path)
    check(len(rows) > 0, f"csv has detection rows ({len(rows)})")
    header = list(rows[0].keys()) if rows else []
    check(header == ["time", "class", "confidence", "x1", "y1", "x2", "y2", "image"],
          f"csv header correct ({header})")
    check(all(r["class"] in ("crack", "hole") for r in rows),
          "only stub classes appear")
    check(FakeModel.seen_imgsz == 320, f"imgsz forwarded to the model ({FakeModel.seen_imgsz})")
    check(FakeModel.seen_conf == 0.4, f"conf forwarded to the model ({FakeModel.seen_conf})")

    pictures = sorted(os.listdir(snap_dir)) if os.path.isdir(snap_dir) else []
    check(len(pictures) > 0, f"detection pictures written ({len(pictures)})")
    img = cv2.imread(os.path.join(snap_dir, pictures[0]))
    check(img is not None, "a detection picture can be opened again")
    linked = [r for r in rows if r["image"]]
    # several boxes in one frame share that frame's picture, so compare the
    # number of distinct pictures, not the number of rows
    distinct = {r["image"] for r in linked}
    check(len(distinct) <= len(pictures),
          f"csv rows reference at most the pictures that exist "
          f"({len(distinct)} referenced, {len(pictures)} written)")
    check(all(os.path.exists(r["image"]) for r in linked),
          "every csv image path points at a real file")

    # one picture per second at most: the stub run is short, so expect few
    check(len(pictures) <= 4, f"snapshot rate is throttled ({len(pictures)} for a short run)")
    return rows


def test_only(tmp, video):
    print("\n== --only filter ==")
    csv_path = os.path.join(tmp, "only.csv")
    run(["--model", "stub", "--src", video, "--port", "0", "--only", "hole",
         "--csv", csv_path, "--no-snap", "--max-frames", "40"])
    rows = read_csv(csv_path)
    check(len(rows) > 0, f"rows written with --only hole ({len(rows)})")
    check(all(r["class"] == "hole" for r in rows),
          "only the requested class is logged")

    rc = run(["--model", "stub", "--src", video, "--port", "0", "--only", "nosuchclass",
              "--csv", os.path.join(tmp, "x.csv")])
    check(rc != 0, "an unknown --only class name is rejected")


def test_stop_on(tmp, video):
    print("\n== --stop-on stop and re-arm ==")
    FakeRobot.events = []
    sys.modules["robot_serial"] = types.SimpleNamespace(Robot=FakeRobot)
    run(["--model", "stub", "--src", video, "--port", "0", "--stop-on", "crack",
         "--csv", os.path.join(tmp, "stop.csv"), "--no-snap", "--max-frames", "40"])
    stops = FakeRobot.events.count("STOP")
    check(stops >= 2,
          f"STOP is re-sent after the defect clears, not latched ({stops} STOPs)")
    check(FakeRobot.events[0] == "connect" and FakeRobot.events[-1] == "close",
          "robot is opened at the start and closed at the end")


def test_stream(tmp, video):
    print("\n== web stream ==")
    port = free_port()
    finished = threading.Event()

    def run_stream():
        # let the detector run a few hundred frames, then act like a Ctrl+C so
        # main() returns and frees the port - a leaked thread would abort the
        # interpreter at exit
        run(["--model", "stub", "--src", video, "--port", str(port),
             "--csv", os.path.join(tmp, "stream.csv"), "--no-snap",
             "--loop-video", "--max-frames", "100000"])
        finished.set()

    thread = threading.Thread(target=run_stream, daemon=True)
    thread.start()

    # wait for the server instead of guessing with a sleep
    base = f"http://127.0.0.1:{port}"
    ready = False
    for _ in range(60):
        try:
            with urllib.request.urlopen(base + "/info.txt", timeout=2) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.25)
    if not ready:
        check(False, "the stream server came up on a free port")
        FakeModel.stop_after = 1        # make the still-running loop exit
        thread.join(timeout=30)
        return
    check(True, "the stream server came up on a free port")

    try:
        with urllib.request.urlopen(base + "/", timeout=10) as r:
            page = r.read().decode(errors="replace")
        check(r.status == 200 and "<img" in page, "web page answers with the video tag")
        check("pipe" in page.lower() or "stream" in page.lower(),
              "web page has the expected content")

        with urllib.request.urlopen(base + "/snapshot.jpg", timeout=10) as r:
            data = r.read()
        check(data[:3] == b"\xff\xd8\xff", "snapshot.jpg is a JPEG")

        with urllib.request.urlopen(base + "/stream.mjpg", timeout=10) as r:
            head = r.read(4096)
        check(b"\xff\xd8\xff" in head, "MJPEG stream carries JPEG frames")

        with urllib.request.urlopen(base + "/info.txt", timeout=10) as r:
            info = r.read().decode(errors="replace")
        check("FPS" in info, f"info line reports the frame rate ({info.strip()})")
    except Exception as exc:
        check(False, f"stream could be reached ({type(exc).__name__}: {exc})")
    finally:
        FakeModel.stop_after = 200        # stop the loop the way Ctrl+C would
        thread.join(timeout=30)
    check(finished.is_set(), "the detector shut its server down and returned")
    check(not thread.is_alive(), "no thread is left running after the test")
    try:                                 # and the port really is free again
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
        check(True, "the port is released after shutdown")
    except OSError as exc:
        check(False, f"the port is released after shutdown ({exc})")


def test_end_of_video(tmp):
    print("\n== video ends ==")
    video = os.path.join(tmp, "short.mp4")
    make_video(video, frames=12)
    rc = run(["--model", "stub", "--src", video, "--port", "0",
              "--csv", os.path.join(tmp, "short.csv"), "--no-snap"])
    check(rc == 0, "run stops cleanly at the end of a video file")

    rc = run(["--model", "stub", "--src", video, "--port", "0", "--loop-video",
              "--max-frames", "40", "--csv", os.path.join(tmp, "loop.csv"),
              "--no-snap"])
    check(rc == 0, "--loop-video keeps going past the end")

    print("\n== Ctrl+C mid-run ==")
    FakeModel.counter = 0
    pi_detect.YOLO = lambda *a, **k: FakeModel()
    FakeModel.stop_after = 25          # act like a keyboard interrupt
    old_argv = sys.argv
    sys.argv = ["pi_detect.py", "--model", MODEL[0], "--src", video, "--port", "0",
                "--csv", os.path.join(tmp, "ctrlc.csv"), "--no-snap",
                "--loop-video", "--max-frames", "100000"]
    try:
        pi_detect.main()
        rc = 0
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.argv = old_argv
        FakeModel.stop_after = 0
    check(rc == 0, "an interrupt mid-run ends the run cleanly")
    check(os.path.exists(os.path.join(tmp, "ctrlc.csv")),
          "the csv is flushed before the run ends")


def main():
    tmp = tempfile.mkdtemp(prefix="pi_detect_test_")
    MODEL[0] = make_model_folder(tmp)
    video = os.path.join(tmp, "clip.mp4")
    make_video(video, frames=60)
    try:
        test_model_path_check(tmp)
        test_csv_and_snapshots(tmp, video)
        test_only(tmp, video)
        test_stop_on(tmp, video)
        test_end_of_video(tmp)
        test_stream(tmp, video)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for message in FAILED:
        print("  failed:", message)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())