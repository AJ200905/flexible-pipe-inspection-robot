"""
cam_utils.py - one Camera class for every input the robot can use.

Part 3 helper (shared with Part 2).  pi_detect.py only ever calls
cam.read() (BGR frame or None) and cam.release().

    Camera("auto")                      # CSI if there is one, else USB webcam
    Camera("usb")                       # force the USB webcam
    Camera("csi")                       # force the Pi camera module (needs picamera2)
    Camera("pipe_run1.mp4")             # a recorded video
    Camera(0)                           # any OpenCV source
    Camera("auto", calib="fisheye_calib.npz")   # undistort every frame
    Camera("run.mp4", loop_video=True)          # restart the clip at the end

The calibration file is the .npz written by Part 2 and holds:
    K : 3x3 camera matrix
    D : distortion coefficients (4 values = fisheye model, anything else =
        the normal pinhole model).  The remap table is built once at start-up,
        which is far cheaper than undistorting frame by frame on a Pi.
"""
import glob
import os
import sys

import cv2
import numpy as np


# --------------------------------------------------------------------- utils
def _have_picamera2():
    try:
        import picamera2  # noqa: F401
        return True
    except Exception:
        return False


def _video_indices():
    """Existing /dev/videoN numbers, lowest first, no duplicates."""
    out = []
    for path in glob.glob("/dev/video*"):
        tail = path[len("/dev/video"):]
        if tail.isdigit():
            out.append(int(tail))
    return sorted(set(out))


def _candidate_indices():
    """Device numbers worth trying: the real ones first, then 0/1/2 as fallback."""
    return _video_indices() + [i for i in (0, 1, 2) if i not in _video_indices()]


def _build_remap(calib_path, width, height):
    """Turn a Part 2 calibration .npz into (map1, map2) for cv2.remap."""
    data = np.load(calib_path)
    keys = set(data.files)
    if not {"K", "D"} <= keys:
        raise ValueError(f"{calib_path} has {sorted(keys)}, need K and D")
    K = np.asarray(data["K"], dtype=np.float64)
    D = np.asarray(data["D"], dtype=np.float64).reshape(-1)
    size = (int(width), int(height))
    if D.size == 4:                       # OpenCV fisheye model
        print(f"[cam] fisheye undistort from {calib_path} @ {size[0]}x{size[1]}")
        return cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, size,
                                                   cv2.CV_32FC1)
    print(f"[cam] pinhole undistort from {calib_path} @ {size[0]}x{size[1]}")
    return cv2.initUndistortRectifyMap(K, D, None, K, size, cv2.CV_32FC1)


# --------------------------------------------------------------------- Camera
class Camera:
    def __init__(self, src="auto", calib=None, loop_video=False,
                 width=640, height=480, fps=30):
        self.src = src
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.loop_video = bool(loop_video)
        self.mode = None                 # "csi" | "usb" | "file" | "stream" | None
        self.path = None
        self._remap = None
        self._frames = 0
        self._cap = None                 # cv2.VideoCapture (usb / file)
        self._picam = None               # picamera2 object (csi)
        self._pending = None             # first frame, kept for the remap

        if calib and not os.path.exists(calib):
            raise FileNotFoundError(f"calibration file not found: {calib}")

        self._open()

        if calib:
            # build the remap from the *first real frame*, not from a property
            # that is often 0.0 - cv2.remap does not complain about a size
            # mismatch, it just quietly returns a wrongly scaled picture.
            try:
                probe = self._grab()
                if probe is None:
                    raise RuntimeError("cannot read a first frame from " + str(self.src))
                self.width, self.height = probe.shape[1], probe.shape[0]
                self._pending = probe
                self._remap = _build_remap(calib, self.width, self.height)
            except Exception:
                self.release()          # never leave the camera open on a bad calib
                raise

    # ------------------------------------------------------------------ open
    def _open(self):
        src = self.src
        if isinstance(src, (int, float)) and not isinstance(src, bool):
            return self._open_usb(int(src))
        src = str(src).strip()
        low = src.lower()               # compare case-insensitively, open the real name

        if low == "auto":
            cap = self._open_csi()
            if cap is not None:
                return cap
            for idx in _candidate_indices():
                cap = self._open_usb(idx, quiet=(idx > 2))
                if cap is not None:
                    return cap
            raise RuntimeError(
                "no camera found. Plug in a USB webcam (or attach the CSI camera) "
                "and check it with:  raspivid /  libcamera-vid  or  v4l2-ctl --list-devices")

        if low == "usb":
            for idx in _candidate_indices():
                cap = self._open_usb(idx, quiet=(idx > 2))
                if cap is not None:
                    return cap
            raise RuntimeError("no USB camera found (tried " + str(_video_indices()) + ")")

        if low == "csi":
            cap = self._open_csi()
            if cap is None:
                raise RuntimeError(
                    "no CSI camera. Install the module:  sudo apt install -y python3-picamera2 "
                    "(a venv with --system-site-packages can then see it)")
            return cap

        # anything else = a file name or an OpenCV URL
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video source: {src}")
        self.mode = "file"
        self.path = src
        self._cap = cap
        print(f"[cam] video file: {src}"
              + ("  (looping)" if self.loop_video else ""))
        return cap

    def _open_usb(self, idx, quiet=False):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        ok, frame = cap.read()
        if not ok or frame is None:            # a metadata node: not a real camera
            cap.release()
            return None
        h, w = frame.shape[:2]
        self.width, self.height = w, h
        self.mode = "usb"
        self.path = idx
        self._cap = cap
        self._pending = frame                 # keep frame 0 for the calibration check
        print(f"[cam] USB camera: /dev/video{idx}  {w}x{h}")
        return cap

    def _open_csi(self):
        """Return a truthy handle when the CSI camera is running, else None."""
        if not _have_picamera2():
            if str(self.src).lower() != "auto":
                raise RuntimeError("picamera2 is not installed "
                                   "(sudo apt install -y python3-picamera2)")
            return None
        try:
            from picamera2 import Picamera2
        except Exception as exc:                # pragma: no cover
            if str(self.src).lower() != "auto":
                raise RuntimeError(f"picamera2 import failed: {exc}")
            return None
        try:
            cam = Picamera2()
            cfg = cam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": self.fps})
            cam.configure(cfg)
            cam.start()
        except Exception as exc:
            if str(self.src).lower() != "auto":
                raise RuntimeError(f"cannot open the CSI camera: {exc}")
            return None
        self.mode = "csi"
        self.path = "csi"
        self._picam = cam
        print(f"[cam] CSI camera: {self.width}x{self.height} @ {self.fps} fps")
        return self._picam          # truthy: frames come from self._picam

    # ------------------------------------------------------------------ read
    def read(self):
        """Next frame as BGR, or None when the source is finished."""
        frame = self._pending if self._pending is not None else self._grab()
        self._pending = None
        if frame is None and self.mode == "file" and self.loop_video:
            frame = self._rewind()          # restart the clip and take its frame 0
        if frame is None:
            return None
        if self._remap is not None:
            if (frame.shape[1], frame.shape[0]) != (self._remap[0].shape[1],
                                                   self._remap[0].shape[0]):
                raise RuntimeError(
                    f"frame is {frame.shape[1]}x{frame.shape[0]} but the calibration "
                    f"was built for {self._remap[0].shape[1]}x{self._remap[0].shape[0]}"
                    " - re-capture with --width/--height matching Part 2")
            frame = cv2.remap(frame, self._remap[0], self._remap[1],
                              cv2.INTER_LINEAR)
        self._frames += 1
        return frame

    def _grab(self):
        if self.mode == "csi":
            if self._picam is None:
                return None                 # already released
            frame = self._picam.capture_array("main")     # RGB888
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def _rewind(self):
        """Jump back to the start of the clip.  Returns frame 0, or None."""
        cap = getattr(self, "_cap", None)
        if cap is None:
            return None
        if cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
        # some codecs refuse to seek: reopen instead
        cap.release()
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self._cap = None
            return None
        self._cap = cap
        ok, frame = cap.read()
        return frame if ok else None

    # ------------------------------------------------------------------ misc
    def size(self):
        cap = getattr(self, "_cap", None)
        if cap is None:
            return self.width, self.height
        return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height))

    def release(self):
        cap = getattr(self, "_cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            self._cap = None
        cam = getattr(self, "_picam", None)
        if cam is not None:
            try:
                cam.stop()
                cam.close()
            except Exception:
                pass
            self._picam = None
        self._pending = None
        self.mode = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


# back-compat: some Part 2 scripts used these names
def open_camera(src="auto", calib=None, loop_video=False, **kw):
    return Camera(src, calib=calib, loop_video=loop_video, **kw)


def default_calib():
    """fisheye_calib.npz next to this file, if it happens to be there."""
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("fisheye_calib.npz", "calib.npz"):
        path = os.path.join(here, name)
        if os.path.exists(path):
            return path
    return None


if __name__ == "__main__":
    # quick manual check:  python3 cam_utils.py auto   or   python3 cam_utils.py run.mp4
    which = sys.argv[1] if len(sys.argv) > 1 else "auto"
    cam = Camera(which, calib=default_calib())
    print("source:", cam.mode, cam.path, "size:", cam.size())
    shown = 0
    try:
        while shown < 5:
            img = cam.read()
            if img is None:
                print("no frame")
                break
            cv2.imshow("cam_utils", img)
            cv2.waitKey(200)
            shown += 1
    finally:
        cam.release()
        cv2.destroyAllWindows()
    print("ok -", shown, "frames read")