#!/usr/bin/env bash
# selftest.sh - check that Part 3 runs on this machine.
#
#   bash selftest.sh                 # no model needed
#   bash selftest.sh best_ncnn_model # also run the real model over a clip
#
# It exercises the whole chain with a stand-in detector: video -> model -> boxes
# -> detections.csv -> pictures -> web stream, plus the option handling.  When
# you pass a model path, it additionally runs your trained NCNN model over a
# synthetic pipe clip and prints the measured speed.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
MODEL_ARG="${1:-}"
WORK="selftest_run"
PY="${PYTHON:-python}"
PASS=0
FAIL=0
SKIP=0

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { PASS=$((PASS+1)); printf '   \033[32mPASS\033[0m %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf '   \033[31mFAIL\033[0m %s\n' "$*"; }
skip() { SKIP=$((SKIP+1)); printf '   \033[33mSKIP\033[0m %s\n' "$*"; }

rm -rf "$WORK"; mkdir -p "$WORK"

say "1. python + packages"
if "$PY" - <<'PY'
from importlib.metadata import version
import cv2, numpy, flask, ultralytics
print("   opencv", cv2.__version__, "| numpy", numpy.__version__,
      "| flask", version("flask"), "| ultralytics", ultralytics.__version__)
try:
    import ncnn
    print("   ncnn", ncnn.__version__, "python bindings present")
except Exception:
    print("   ncnn bindings not importable - the ultralytics ncnn backend may still work")
try:
    import picamera2
    print("   picamera2 present (CSI cameras supported)")
except Exception:
    print("   no picamera2 - USB cameras and video files only")
PY
then ok "packages import"
else
    bad "imports failed"
    echo "   libGL.so.1 missing?  sudo apt install -y libgl1-mesa-gl"
    echo "   or:                  pip install opencv-python-headless"
fi

say "2. synthetic test video"
if "$PY" make_test_video.py "$WORK/test_tunnel.mp4" --seconds 4 >/dev/null 2>&1 \
   && [ -s "$WORK/test_tunnel.mp4" ]; then
    ok "make_test_video.py wrote $(ls -la "$WORK"/test_tunnel*.mp4 | wc -l) clips"
else
    bad "make_test_video.py failed"
fi

say "3. inference engines on this CPU"
if timeout 600 "$PY" ncnn_check.py > "$WORK/ncnn.log" 2>&1; then
    sed 's/^/   /' "$WORK/ncnn.log"
    ok "NCNN loads and runs here"
else
    sed 's/^/   /' "$WORK/ncnn.log"
    if grep -q "could not download" "$WORK/ncnn.log"; then
        skip "could not download the test network (no internet?) - run ncnn_check.py later"
    else
        bad "ncnn_check.py failed"
    fi
fi

say "4. pipeline test (stand-in detector)"
# test_pi_detect.py drives pi_detect.py itself with a fake model: csv, snapshots,
# --only, --stop-on stop/re-arm, video end, --loop-video and the web stream
if timeout 900 "$PY" -u test_pi_detect.py > "$WORK/pipeline.log" 2>&1; then
    RC=0
else
    RC=$?
fi
RESULT=$(grep -E "^[0-9]+ passed" "$WORK/pipeline.log" | tail -1)
grep -E "^   FAIL" "$WORK/pipeline.log" | sed 's/^/  /'
echo "   summary: ${RESULT:-no result}"
if [ "$RC" = "0" ] && [ "${RESULT#*0 failed}" != "$RESULT" ]; then
    ok "pi_detect.py pipeline behaves ($RESULT)"
else
    bad "pipeline test failed (${RESULT:-no result}, exit $RC, see $WORK/pipeline.log)"
fi

say "5. argument and error handling"
"$PY" pi_detect.py --model definitely_not_here --list-classes >/dev/null 2>&1
[ $? -ne 0 ] && ok "missing model is rejected" || bad "missing model not caught"
"$PY" pi_detect.py --conf 3 --list-classes >/dev/null 2>&1
[ $? -ne 0 ] && ok "impossible --conf is rejected" || bad "bad --conf not caught"
"$PY" pi_detect.py --help >/dev/null 2>&1
[ $? -eq 0 ] && ok "--help works" || bad "--help failed"

say "6. camera"
if "$PY" - <<'PY' 2>/dev/null
from cam_utils import Camera
try:
    cam = Camera("auto")
    frame = cam.read()
    print("   opened %s source, frame %s" % (cam.mode, None if frame is None else frame.shape))
    cam.release()
except Exception as exc:
    print("   no camera available (%s)" % str(exc)[:110])
    raise SystemExit(3)
PY
then
    ok "a camera could be opened"
else
    skip "no camera plugged in (USB or CSI) - everything else was still tested"
fi

say "7. real model (optional)"
if [ -z "$MODEL_ARG" ]; then
    skip "no model path given - run 'bash selftest.sh best_ncnn_model' to include it"
else
    if [ -d "$MODEL_ARG" ]; then
        ok "model folder exists: $MODEL_ARG"
    else
        bad "model not found: $MODEL_ARG"
    fi
    "$PY" pi_detect.py --model "$MODEL_ARG" --list-classes 2>&1 | head -2 | sed 's/^/   /'
    "$PY" pi_detect.py --model "$MODEL_ARG" --src "$WORK/test_tunnel_defects.mp4" \
        --port 0 --csv "$WORK/model.csv" --snap-dir "$WORK/model_snaps" \
        --max-frames 60 > "$WORK/model.log" 2>&1
    tail -4 "$WORK/model.log" | sed 's/^/   /'
    if [ -f "$WORK/model.csv" ]; then
        ok "the real model ran over the clip ($(($(wc -l < "$WORK/model.csv") - 1)) detection rows)"
    else
        bad "the real model produced no csv"
    fi
fi

echo
if [ "$FAIL" -eq 0 ]; then
    printf '\033[32m%d checks passed, 0 failed\033[0m' "$PASS"
else
    printf '\033[31m%d checks passed, %d failed\033[0m' "$PASS" "$FAIL"
fi
[ "$SKIP" -gt 0 ] && printf ", %d skipped" "$SKIP"
printf '  (files kept in %s/)\n' "$WORK"
exit $((FAIL > 0))