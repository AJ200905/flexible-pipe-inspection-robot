# Pipe Inspection Robot - Part 3: crack detection with YOLO26n

Train on Google Colab (GPU), export to NCNN at 320x320, run on the Raspberry Pi 4
(CPU only).

```
LAPTOP / COLAB (GPU)                        RASPBERRY PI 4 (CPU only)
Roboflow dataset -> train YOLO26n -> export NCNN -> pi_detect.py -> browser view
(pipe defects)      (320x320)     best_ncnn_model      camera/video      + detections.csv
```

* **Why Colab** - training needs a GPU; the Pi 4 has none usable for AI.
* **Why YOLO26n** - the nano model is small enough for a Pi CPU.
* **Why NCNN** - built for ARM CPUs, far faster than the PyTorch `.pt` file.
* **Why 320x320** - about 4x less work than 640x640; close-up cracks stay big enough.

## Files

| File | Where | What it does |
|---|---|---|
| `colab/pipe_defects_yolo26n_train.ipynb` | Google Colab | download data, train, test, export NCNN, download |
| `pi/pi_detect.py` | Pi (or laptop) | run the model on video or live camera, stream + log |
| `pi/cam_utils.py` | Pi | one `Camera` class for `auto` / `usb` / `csi` / a video file, plus fisheye undistortion |
| `pi/stream_server.py` | Pi | the MJPEG web page on port 5000 |
| `pi/robot_serial.py` | Pi | optional ESP32 link used by `--stop-on` |
| `pi/make_test_video.py` | Pi | fake pipe-tunnel clips for testing before real footage exists |
| `pi/ncnn_check.py` | Pi | check NCNN works on this CPU and measure ms per frame |
| `pi/test_pi_detect.py` | Pi | 24 checks of pi_detect.py itself (no trained model needed) |
| `pi/selftest.sh` | Pi | run every check at once |
| `install.sh`, `requirements.txt` | Pi | one-time setup |

## The dataset

**Pipe Defects Detection** - Roboflow Universe, user `gavril-birgauan`,
<https://universe.roboflow.com/gavril-birgauan/pipe-defects-detection>.
Around 1,000 original CCTV images of sewer pipes (version 1 = 2,406 after
Roboflow augmentation). Seven classes: `crack, hole, debris, buckling, jntoffs,
obsc, utits` - the short names are the dataset author's abbreviations, so open a
few examples of each before you describe them in your report.
Licence **CC BY 4.0**: free to use, credit the author.

> These are sewer-pipe CCTV frames. Acrylic and metal pipes look different
> (shine, colour, transparency), so expect lower accuracy on your own pipes
> until you add your own labelled pictures - see *Make it work on YOUR pipes*.

## 1. Train (Colab)

1. <https://colab.research.google.com> -> *File -> Upload notebook* ->
   `colab/pipe_defects_yolo26n_train.ipynb`.
2. *Runtime -> Change runtime type -> T4 GPU -> Save*.
3. Get the key: roboflow.com -> *Settings -> API Keys -> Private API Key*.
4. Open the Universe dataset page, **Fork Dataset** into your workspace, then
   **Download Dataset** in *YOLOv8/YOLO11* format.
5. Run the cells top to bottom (**Shift+Enter**). Paste the key at section 4 and
   put your own workspace name in `WORKSPACE`.

| # | Section | What happens |
|---|---|---|
| 1 | Check GPU | prints Tesla T4; if not, the runtime type is wrong |
| 2 | Install | ultralytics, roboflow, ncnn, pnnx (~1 min) |
| 3 | Google Drive | click *Allow*; results are kept in Drive |
| 4 | Download dataset | paste the Roboflow key |
| 5 | Fix data.yaml | absolute paths; makes a test split if there is none |
| 6 | Look at data | boxes per class, 6 labelled pictures |
| 7 | Settings | `yolo26n.pt`, 320 px, 100 epochs, batch 64, patience 20 |
| 8 | Train | ~20-40 min; watch mAP50 grow |
| 8b | Resume | only after a disconnect (re-run 2 and 3 first) |
| 9 | Curves | loss should fall and flatten; check the confusion matrix |
| 10 | Test score | mAP50 overall and per class |
| 11 | Predictions | 6 test pictures with the model's boxes |
| 12 | Export NCNN | creates `best_ncnn_model/` at 320 px |
| 13 | NCNN check | CPU milliseconds per image |
| 14 | Download | `best_ncnn_model.zip` (for the Pi) and `best.pt` (keep) |

The four important lines:

```python
model = YOLO('yolo26n.pt')                            # pretrained nano model
model.train(data=DATA_YAML, imgsz=320, epochs=100, batch=64, patience=20)
best = YOLO(BEST)                                     # best epoch
best.export(format='ncnn', imgsz=320)                 # -> best_ncnn_model/
```

Reading the results:

| Number | Meaning | Rough guide |
|---|---|---|
| mAP50 | how well the boxes match the true defects (0-1) | > 0.5 usable, > 0.7 good |
| Precision | of the boxes drawn, how many were right | low = too many false alarms, raise `--conf` |
| Recall | of the real defects, how many were found | low = missed defects, lower `--conf`, add data |
| Confusion matrix | which classes get mixed up | big off-diagonal squares = confusion |

## 2. Set the Pi up (once)

```bash
uname -m                       # must print aarch64
bash install.sh                # venv + packages + camera check
source ~/robot-env/bin/activate     # needed in every new SSH window
```

`install.sh` uses `python3 -m venv --system-site-packages` so the venv can still
see the system camera libraries, and falls back to `virtualenv.pyz` if
`python3-venv` is not installed. It also installs `opencv-python-headless`,
because a Pi without a desktop has no `libGL` and plain `opencv-python` fails to
import with `ImportError: libGL.so.1`.

Then prove the whole chain works, with no model and no camera needed:

```bash
cd ~/pipe_robot/pi
bash selftest.sh                       # every check
python3 ncnn_check.py                  # NCNN speed on this CPU
python3 test_pi_detect.py              # just the pipeline checks
```

## 3. Copy the model and test on a recorded video

```bash
# on the laptop
scp best_ncnn_model.zip jellyfish@JellyFish.local:~/pipe_robot/pi/
scp pipe_run1.mp4      jellyfish@JellyFish.local:~/pipe_robot/pi/

# on the Pi
cd ~/pipe_robot/pi
unzip best_ncnn_model.zip -d best_ncnn_model
source ~/robot-env/bin/activate
python3 pi_detect.py --model best_ncnn_model --src pipe_run1.mp4
```

The zip holds `model.ncnn.param` and `model.ncnn.bin`, and the `-d best_ncnn_model`
puts them in the folder name `pi_detect.py` looks for. Sanity check any time
with `python3 pi_detect.py --model best_ncnn_model --list-classes`.

Open <http://jellyfish.local:5000> on the laptop to watch the boxes and FPS.
Always test on a recorded video first: you can repeat the exact same run after
each change and compare. Live input changes every time.

No footage yet? Generate a clip to prove the plumbing works:

```bash
python3 make_test_video.py test_tunnel.mp4 --seconds 10
python3 pi_detect.py --model best_ncnn_model --src test_tunnel.mp4 --max-frames 100
```

## 4. Run live

```bash
python3 pi_detect.py --model best_ncnn_model --src auto

# with fisheye correction, saving the annotated video
python3 pi_detect.py --model best_ncnn_model --src auto --calib fisheye_calib.npz --save run1.mp4

# only look for cracks and holes
python3 pi_detect.py --model best_ncnn_model --src auto --only crack hole

# stop the robot when a crack is seen (ESP32 on USB serial)
python3 pi_detect.py --model best_ncnn_model --src auto --stop-on crack
```

| Option | Default | Meaning |
|---|---|---|
| `--model` | `best_ncnn_model` | NCNN folder (or `best.pt` on the laptop) |
| `--src` | `auto` | `auto` / `usb` / `csi` / a video file |
| `--imgsz` | 320 | keep it the same as training |
| `--conf` | 0.40 | minimum confidence; raise for fewer false alarms |
| `--only` | all classes | e.g. `--only crack` |
| `--calib` | none | fisheye calibration `.npz` from Part 2 |
| `--save` | none | save the annotated video |
| `--stop-on` | none | send STOP to the ESP32 when these classes appear |
| `--port` | 5000 | web stream port |
| `--csv` | `detections.csv` | log file |
| `--max-frames` | 0 | stop after N frames (0 = run until Ctrl+C) |
| `--list-classes` | | print the model's classes and exit |

## Outputs

* `detections.csv` - every detection: time, class, confidence, box corners, picture name.
* `detections/` - a picture of the frame each second something is found (max one per second).
* Expected speed on a Pi 4 at 320 px: roughly 4-8 FPS (the stream shows the real
  number). That is fine for a slow robot - drive at low speed while inspecting.

`--stop-on` sends `STOP` once per defect event and prints when the robot has been
re-armed; it does **not** send `RESET`, so the driver stays disabled until you
reset it by hand (`python3 robot_serial.py /dev/ttyUSB0` then `reset()`).

## Make it work on YOUR pipes

1. Record videos inside your acrylic and metal pipes (Part 2, Step 5), with and
   without cracks.
2. On the dataset page, **Fork Dataset** to get a copy in your workspace.
3. Upload your videos (Roboflow can cut them into frames, e.g. 2 per second).
4. Annotate: draw a box around each crack. 100-300 of your own images make a big
   difference.
5. Generate a new version (resize 320, add flips/brightness augmentation).
6. In notebook section 4 set `WORKSPACE`, `PROJECT`, `VERSION` to yours and run
   everything again.

Fine-tuning shortcut: in section 7 set `MODEL = "/content/best.pt"` (upload your
previous best) instead of `yolo26n.pt`, with fewer epochs (e.g. 50). It starts
from what it already learned.

## Troubleshooting

| Problem | Fix |
|---|---|
| Colab: CUDA not available | *Runtime -> Change runtime type -> T4 GPU*, rerun from cell 1 |
| Colab: out of memory | lower `BATCH` to 32 or 16 |
| Colab disconnected | run cells 2, 3, 7 then the Resume cell (8b) |
| Export error about pnnx | rerun the install cell (it installs ncnn and pnnx) |
| Pi: `pip install ultralytics` fails | check `uname -m` is `aarch64`, upgrade pip, use the venv |
| Pi: `ImportError: libGL.so.1` | `pip install opencv-python-headless` (already in requirements) or `sudo apt install -y libgl1-mesa-gl` |
| Pi: `Illegal instruction` (SIGILL) as soon as a `.pt` model runs | expected on a Pi 4 CPU: the PyPI PyTorch wheel needs ARMv8.2 instructions the Pi 4 does not have. **Use the NCNN folder** - `python3 ncnn_check.py` shows whether NCNN works on your machine |
| Pi: "does not look like an NCNN model" | unzip again with `-d best_ncnn_model`, so the folder holds `model.ncnn.param` and `model.ncnn.bin` |
| Pi: no web view, "port already in use" | another copy is running; use `--port 0` to detect without the stream |
| Pi: very slow (< 2 FPS) | load the NCNN folder, not `best.pt`; close other programs; check `--imgsz 320` |
| Lots of false boxes | raise `--conf` to 0.5-0.6; add your own pipe images |
| Misses real cracks | lower `--conf` to 0.25-0.3; better lighting; more labelled images |
| `--stop-on` cannot find the ESP32 | `ls /dev/ttyUSB* /dev/ttyACM*`, then `--robot-port /dev/ttyUSB0` |
| `no camera found` | plug the USB webcam in; `v4l2-ctl --list-devices`; for CSI: `sudo apt install python3-picamera2` |

### Speed on a Pi 4 at 320 px

`python3 ncnn_check.py` measures the network only on your own Pi. A stock
YOLO11n at 320x320 measures around 110-140 ms per frame here, i.e. a ceiling of
roughly 7-9 FPS, which matches the 4-8 FPS you should see in `pi_detect.py`
once capture, drawing and streaming are added. That is plenty for a slow robot -
drive at low speed while inspecting.