"""Thread-safe serial link to the ESP32 firmware (pipe_robot_esp32.ino)."""

import glob
import threading
import time

import serial

TELEMETRY_KEYS = ["roll", "pitch", "cmd1", "cmd2", "cmd3",
                  "rpm1", "rpm2", "rpm3", "light", "servo"]


def find_port():
    ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    return ports[0] if ports else None


class Esp32Link:
    """Opens the port, reads telemetry in a background thread, sends commands."""

    def __init__(self, port="auto", baud=115200, on_message=None):
        self.port = find_port() if port in ("", "auto") else port
        if not self.port:
            raise RuntimeError("ESP32 not found (/dev/ttyUSB* or /dev/ttyACM*)")
        self.ser = serial.Serial(self.port, baud, timeout=0.1)
        self.on_message = on_message
        self.telemetry = None          # latest dict
        self.telemetry_time = 0.0
        self._lock = threading.Lock()
        self._running = True
        time.sleep(2.0)                # ESP32 resets when the port opens
        self.ser.reset_input_buffer()
        threading.Thread(target=self._reader, daemon=True).start()

    @property
    def alive(self):
        return self._running

    def send(self, text):
        try:
            with self._lock:
                self.ser.write((text.strip() + "\n").encode())
        except (serial.SerialException, OSError):
            self._running = False

    def _reader(self):
        while self._running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
            except (serial.SerialException, OSError):
                self._running = False
                break
            if not line:
                continue
            if line.startswith("T,"):
                parts = line[2:].split(",")
                if len(parts) == len(TELEMETRY_KEYS):
                    try:
                        self.telemetry = dict(zip(TELEMETRY_KEYS, map(float, parts)))
                        self.telemetry_time = time.time()
                    except ValueError:
                        pass
            elif self.on_message:
                self.on_message(line)

    def close(self):
        try:
            self.send("S")
            time.sleep(0.05)
        finally:
            self._running = False
            try:
                self.ser.close()
            except Exception:
                pass
