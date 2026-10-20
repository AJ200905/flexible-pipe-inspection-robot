# Flexible Pipe Inspection Robot

TDCR-based flexible pipe inspection robot — autonomous pipeline inspection with a
Tendon-Driven Continuum Robot (TCR) link, vision-based defect detection, and
digital path mapping.

Project 1, School of Mechanical Engineering, VIT Chennai.

## Repository layout

| Path | Contents |
|---|---|
| `pipe_robot/` | Raspberry Pi 4 defect-detection pipeline: YOLO26n trained on Colab, exported to NCNN, live MJPEG stream, CSV logging, optional ESP32 link. See `pipe_robot/README.md`. |
| `firmware/v3/esp32/` | Current ESP32 firmware (v3): 3 DC drive motors on 2× TB6612FNG, WS2812 LED, Goteck GD2508MG tendon servo, MPU6050 IMU, USB-serial command/telemetry protocol. |
| `firmware/v3/ros2_ws/` | ROS 2 package `pipe_robot_ros`: serial bridge, ESP32 link, vision node, launch files, RViz config. |
| `firmware/tendon_update/` | Tendon-control firmware variant and its matching ROS 2 package. |
| `arduino/` | Bench-test sketches: `servo_test` (pulse sweep on GPIO 23) and `center_servo` (hold midpoint). |
| `scripts/check_pi.sh` | Raspberry Pi environment/diagnostic check. |
| `docs/pi_report.txt` | Captured Pi system report (OS, network, memory, storage, camera). |
| `Pipe_Inspection_Robot_Review1_v2.pptx` | Project review presentation. |

## System overview

```
        ESP32 (firmware/v3)                     Raspberry Pi 4
  motors · tendon servo · IMU  <-- USB --->  YOLO26n/NCNN detection
        LED · telemetry            115200      MJPEG stream + detections.csv
```

* **Drive:** three DC motors through two TB6612FNG drivers.
* **Steering:** Goteck GD2508MG servo pulling a tendon to bend the TPU continuum link.
* **Sensing:** MPU6050 IMU on the ESP32; USB camera / CSI camera on the Pi.
* **Perception:** YOLO26n nano model, 320×320 input, exported to NCNN for CPU inference.
* **Safety:** motors stop if no serial command arrives for 1 second; tendon bend is held.

## Quick start

### Raspberry Pi — detection pipeline

```bash
cd pipe_robot
bash install.sh
source ~/robot-env/bin/activate
cd pi
bash selftest.sh
```

Training happens on Google Colab; see `pipe_robot/README.md` and
`pipe_robot/colab/pipe_defects_yolo26n_train.ipynb`.

### ESP32 — firmware

Open `firmware/v3/esp32/pipe_robot_esp32/pipe_robot_esp32.ino` in the Arduino IDE,
select **ESP32 Dev Module**, install the **Adafruit NeoPixel** library, and flash.

### ROS 2 — serial/vision nodes

```bash
cd firmware/v3/ros2_ws
colcon build
source install/setup.bash
ros2 launch pipe_robot_ros robot.launch.py
```

## Not tracked in this repository

These are generated or too large for Git and are recreated locally:

* `pipe_robot/detections/` — captured detection frames (~440 MB).
* `pipe_robot/best_ncnn_model/` and `best_ncnn_model.zip` — trained model export.
* `*.pt`, `*.mp4`, dataset archives, ROS 2 `build/`, `install/`, `log/`, `__pycache__/`.

To recreate the model, run the Colab notebook and export `best_ncnn_model.zip`
to the Pi as described in `pipe_robot/README.md`.

## License

See [LICENSE](LICENSE).
