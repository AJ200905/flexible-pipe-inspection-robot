#!/usr/bin/env bash
# One-time setup for Part 3 on the Raspberry Pi.
#   bash install.sh
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/robot-env}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== 1. check the machine is 64-bit =="
ARCH="$(uname -m)"
echo "uname -m = $ARCH"
if [ "$ARCH" != "aarch64" ]; then
    echo "!! Expected aarch64 (64-bit Raspberry Pi OS). 32-bit cannot run this." >&2
    exit 1
fi

echo
echo "== 2. python environment in $ENV_DIR =="
if [ ! -x "$ENV_DIR/bin/python" ]; then
    if python3 -m venv --system-site-packages "$ENV_DIR" 2>/dev/null; then
        echo "created with venv"
    else
        # Raspberry Pi OS Lite / a python without ensurepip: no sudo needed
        # because virtualenv ships its own pip.
        echo "python3-venv missing, using virtualenv.pyz instead"
        curl -fsSL -o /tmp/virtualenv.pyz https://bootstrap.pypa.io/virtualenv.pyz || {
            echo "could not download virtualenv.pyz - check the internet" >&2
            exit 1
        }
        python3 /tmp/virtualenv.pyz --system-site-packages "$ENV_DIR"
    fi
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo
echo "== 3. python packages (this takes 10-20 min on a Pi 4) =="
pip install -r "$HERE/requirements.txt"

echo
echo "== 4. camera libraries =="
python - <<'PY'
try:
    import cv2
    print("opencv", cv2.__version__)
except Exception as exc:
    print("!! opencv missing:", exc)
try:
    import picamera2
    print("picamera2 ok - CSI cameras will work")
except Exception:
    print("no picamera2: only USB cameras will work.")
    print("  (for the CSI module:  sudo apt install -y python3-picamera2 )")
PY

echo
echo "== 5. can this machine run the model? =="
(cd "$HERE/pi" && python ncnn_check.py) || true

echo
echo "== 6. ready =="
cat <<'TXT'
Next:
  1. copy the model over from your laptop
       scp best_ncnn_model.zip <user>@<pi>:~/pipe_robot/pi/
  2. unzip it
       cd ~/pipe_robot/pi && unzip best_ncnn_model.zip -d best_ncnn_model
  3. always activate the environment first
       source ~/robot-env/bin/activate
  4. prove everything works, no model needed
       cd ~/pipe_robot/pi && bash selftest.sh
  5. test on a recorded video, then go live
       python3 pi_detect.py --model best_ncnn_model --src pipe_run1.mp4
       python3 pi_detect.py --model best_ncnn_model --src auto
  6. watch it on the laptop
       http://<pi-hostname>.local:5000
TXT