#!/bin/bash
# Pipe robot - Pi status report (read-only, changes nothing)
s(){ echo; echo "===== $1 ====="; }
have(){ command -v "$1" >/dev/null 2>&1; }

s "SYSTEM"
. /etc/os-release; echo "OS: $PRETTY_NAME"
echo "Kernel: $(uname -r)   Arch: $(uname -m)"
echo "Board: $(tr -d '\0' 2>/dev/null < /proc/device-tree/model)"
echo "Hostname: $(hostname)   User: $(whoami)"
echo "Groups: $(id -nG)"
echo "Uptime: $(uptime -p)"
awk '{printf "CPU temp: %.1f C\n",$1/1000}' /sys/class/thermal/thermal_zone0/temp 2>/dev/null
have vcgencmd && echo "Throttled: $(vcgencmd get_throttled)"
free -h | head -2
df -h / | tail -1

s "NETWORK"
echo "IPs: $(hostname -I)"
have ip && ip -br addr | grep -v '^lo'
for svc in ssh avahi-daemon chrony; do printf "%-13s %s\n" "$svc" "$(systemctl is-active $svc 2>/dev/null)"; done

s "ROS 2"
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
  echo "ROS_DISTRO=$ROS_DISTRO"
  echo "Installed ros-humble packages: $(dpkg-query -W 'ros-humble-*' 2>/dev/null | wc -l)"
  dpkg-query -W -f='${Package} ${Version}\n' 'ros-humble-*' 2>/dev/null | grep -E \
   'ros-humble-(ros-base|desktop|ros-core|cv-bridge|image-transport|compressed-image-transport|vision-msgs|v4l2-camera|usb-cam|camera-ros|robot-state-publisher|xacro|joint-state-publisher|tf2-ros|rmw-|demo-nodes-py|imu-|micro-ros|rosbridge)' \
   | sed 's/^/  /'
else
  echo "ROS 2 Humble NOT found in /opt/ros/humble"; ls /opt/ros 2>/dev/null
fi
echo "-- ROS lines in ~/.bashrc:"
grep -nE 'ros|ROS_|RMW_|venv|robot-env' ~/.bashrc 2>/dev/null | sed 's/^/  /'
echo "-- ROS workspaces / packages (package.xml):"
find ~ -maxdepth 5 -name package.xml -not -path '*/.*' 2>/dev/null | sed 's/^/  /'

s "PYTHON"
python3 --version
check_py(){ timeout 180 "$1" - <<'PY'
import importlib, sys, warnings; warnings.filterwarnings('ignore')
print("python:", sys.executable, sys.version.split()[0])
for m in ["rclpy","cv2","numpy","serial","flask","yaml","cv_bridge","picamera2","ultralytics","ncnn","torch","onnxruntime"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK  {m:12s} {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  --  {m:12s} {type(e).__name__}: {str(e)[:70]}")
PY
}
check_py python3
for v in $(ls -d ~/robot-env ~/*env* ~/.venv 2>/dev/null | sort -u); do
  [ -x "$v/bin/python" ] && { echo; echo "-- venv: $v"; check_py "$v/bin/python"; }
done

s "USB / SERIAL (ESP32)"
have lsusb && lsusb
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "No ESP32 serial port (/dev/ttyUSB* or /dev/ttyACM*)"

s "CAMERA"
v=$(ls /dev/video* 2>/dev/null | paste -sd' '); echo "${v:-No /dev/video* devices}"
have v4l2-ctl && v4l2-ctl --list-devices 2>/dev/null
have rpicam-hello && rpicam-hello --list-cameras 2>&1 | head -6
have libcamera-hello && libcamera-hello --list-cameras 2>&1 | head -6
grep -nE 'camera|dtoverlay|start_x|dtparam=i2c' /boot/firmware/config.txt 2>/dev/null | sed 's/^/  config.txt:/'

s "I2C"
ls /dev/i2c-* 2>/dev/null || echo "No /dev/i2c-* (only matters if IMU is on the Pi; on our design it's on the ESP32)"

s "TOOLS"
for t in tmux git colcon rosdep docker arduino-cli v4l2-ctl gdown chronyc; do
  printf "  %-12s %s\n" "$t" "$(command -v $t || echo MISSING)"
done

s "PROJECT FILES"
ls -la ~/pipe_robot 2>/dev/null || echo "~/pipe_robot not found"
echo "-- models / firmware found:"
find ~ -maxdepth 5 \( -name '*.ncnn.param' -o -name 'best*.pt' -o -name '*.ino' -o -name 'robot_serial.py' -o -name 'pi_detect.py' \) -not -path '*/.*' 2>/dev/null | sed 's/^/  /'

s "AUTOSTART SERVICES"
systemctl list-unit-files --state=enabled 2>/dev/null | grep -iE 'robot|ros|pipe|camera' || echo "none"

echo; echo "===== END ====="
