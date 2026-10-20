#!/usr/bin/env python3
"""
ncnn_check.py - is NCNN actually working on this machine, and how fast?

Useful right after install.sh: it loads a real NCNN network (downloaded once,
then cached in ~/.ncnn_check/) and runs it on a synthetic 320x320 frame.

    python3 ncnn_check.py

Why this matters on a Raspberry Pi: PyTorch from PyPI can crash with
"Illegal instruction" on a Pi 4 CPU (ARMv8.0, no dot-product unit), which makes
best.pt unusable.  NCNN has its own kernels and normally runs fine - this script
tells you which situation you are in.

Note: the network used here is a stock COCO YOLO11n, not a pipe-defect model.
It only measures speed, it cannot find cracks.
"""
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

BASE = "https://github.com/nihui/ncnn-assets/raw/master/models/"
CACHE = os.path.join(os.path.expanduser("~"), ".ncnn_check")
FILES = {"yolo11n.ncnn.param": "net.param", "yolo11n.ncnn.bin": "net.bin"}


def fetch():
    os.makedirs(CACHE, exist_ok=True)
    for name, local in FILES.items():
        dest = os.path.join(CACHE, local)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        print(f"downloading {name} ...", flush=True)
        try:
            with urllib.request.urlopen(BASE + name, timeout=300) as response:
                data = response.read()
        except Exception as exc:
            print(f"  could not download {name}: {exc}")
            print("  check the connection and run this again, or drop the files into")
            print(f"  {CACHE}/ by hand")
            raise SystemExit(1)
        with open(dest, "wb") as fh:
            fh.write(data)


def torch_probe():
    """Can PyTorch run a convolution here? (it cannot on some Pi CPU builds).

    Runs in a subprocess on purpose: an "Illegal instruction" crash would
    otherwise take this script down with it and hide the NCNN result.
    """
    code = (
        "import torch;"
        "torch.nn.Conv2d(3, 8, 3)(torch.zeros(1, 3, 64, 64)).sum().item();"
        "print('works')"
    )
    try:
        done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, timeout=300)
    except Exception as exc:
        return f"not installed ({exc})"
    if done.returncode == 0 and "works" in done.stdout:
        return "works"
    if done.returncode < 0:
        import signal
        try:
            name = signal.Signals(-done.returncode).name
        except ValueError:
            name = f"signal {-done.returncode}"
        return f"crashes ({name}) - a .pt model will not run here"
    return f"fails (exit {done.returncode}): {done.stderr.strip()[-120:]}"


def main():
    print("== PyTorch on this machine ==")
    print("   ", torch_probe())

    print("\n== NCNN ==")
    try:
        import ncnn
    except Exception as exc:
        print("    ncnn is not installed:", exc)
        print("    fix with:  pip install ncnn")
        raise SystemExit(1)
    print("    ncnn version:", ncnn.__version__)

    fetch()
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = max(1, os.cpu_count() or 1)
    if net.load_param(os.path.join(CACHE, "net.param")) != 0:
        raise SystemExit("could not load the param file")
    if net.load_model(os.path.join(CACHE, "net.bin")) != 0:
        raise SystemExit("could not load the bin file")
    print("    network loaded from", CACHE)

    rng = np.random.default_rng(0)
    frame = rng.random((3, 320, 320)).astype(np.float32)
    names = net.input_names()
    outputs = sorted(net.output_names())

    def once():
        with net.create_extractor() as ex:
            ex.input(names[0], ncnn.Mat(frame))
            return [np.array(ex.extract(name)[1]) for name in outputs]

    first = once()                       # warm-up
    print("    input blob :", names, frame.shape)
    for name, arr in zip(outputs, first):
        print(f"    output blob: {name} {arr.shape}")

    runs = 10
    t0 = time.time()
    for _ in range(runs):
        once()
    ms = (time.time() - t0) / runs * 1000
    print("\n    %.1f ms per 320x320 image -> %.1f FPS ceiling (network only)"
          % (ms, 1000 / max(ms, 1e-6)))
    print("\nNCNN RUNS ON THIS MACHINE. A trained pipe-defect model of similar size")
    print("should give a similar per-frame cost inside pi_detect.py.")
    print("This stock model is a COCO detector - it cannot find cracks.")


if __name__ == "__main__":
    main()