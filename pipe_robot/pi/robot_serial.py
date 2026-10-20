"""
robot_serial.py - talk to the ESP32 motor board over the USB serial link.

Only used when pi_detect.py is started with --stop-on, e.g.

    python3 pi_detect.py --model best_ncnn_model --src auto --stop-on crack

The protocol is deliberately plain text, one command per line:

    -> STOP            motor driver disabled until RESET
    -> RESET           re-arm the driver (robot may move again)
    -> SPEED 30        0..100, forward
    -> REVERSE 30      0..100, backward
    <- OK / ERR ...

Wire the ESP32 USB to the Pi, find the port with  ls /dev/ttyUSB* /dev/ttyACM*,
and if needed pass it explicitly:  Robot("/dev/ttyUSB0").

If pyserial is missing:  pip install pyserial
"""
import glob
import time


def find_port(hint=None):
    """First plausible USB-serial port, or None.

    /dev/serial0 is deliberately *not* searched: that is the Pi's own GPIO
    UART, and an ESP32 on the USB port shows up as /dev/ttyUSB* or /dev/ttyACM*.
    Pass it explicitly with --robot-port if you really wired it that way.
    """
    if hint:
        return hint
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    return None


class Robot:
    def __init__(self, port=None, baud=115200, timeout=0.5, dry_run=False):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is not installed (pip install pyserial) - needed for --stop-on"
            ) from exc

        self.port = find_port(port)
        if self.port is None:
            raise RuntimeError(
                "no ESP32 USB serial port found. Check the USB cable, then look for\n"
                "  ls /dev/ttyUSB* /dev/ttyACM*\n"
                "and pass the port, e.g.  --robot-port /dev/ttyUSB0")

        self.dry_run = dry_run
        self.last_sent = None
        self.port_name = self.port
        if dry_run:
            print(f"[robot] DRY RUN - would use {self.port} at {baud} baud")
            self.ser = None
            return

        try:
            self.ser = serial.Serial(self.port, baud, timeout=timeout)
        except Exception as exc:
            raise RuntimeError(f"cannot open {self.port}: {exc}") from exc

        # start from a known-safe state: the driver is disabled, not armed
        self.ser.reset_input_buffer()
        self.stop()
        print(f"[robot] connected: {self.port} @ {baud} baud "
              "(driver stopped; drive commands will not move until the robot is armed)")

    # ------------------------------------------------------------------ io
    def send(self, command, expect_reply=True):
        """Send one command; return the reply string (or None)."""
        self.last_sent = command
        if self.dry_run:
            print(f"[robot] DRY RUN -> {command}")
            return "OK"
        self.ser.write((command + "\n").encode())
        self.ser.flush()
        if not expect_reply:
            return None
        deadline = time.time() + 0.6
        while time.time() < deadline:
            line = self.ser.readline()
            if line:
                return line.decode(errors="replace").strip()
        return None

    # -------------------------------------------------------------- commands
    def stop(self):
        """Cut motor power.  Safe to call every frame."""
        return self.send("STOP")

    def reset(self):
        """Re-arm the driver so the robot can be driven again."""
        return self.send("RESET")

    def speed(self, percent):
        return self.send(f"SPEED {int(percent)}")

    def reverse(self, percent):
        return self.send(f"REVERSE {int(percent)}")

    def close(self):
        if self.dry_run:
            return
        if getattr(self, "ser", None) is not None:
            try:
                self.stop()
                time.sleep(0.05)
                self.ser.close()
            except Exception:
                pass
            self.ser = None


if __name__ == "__main__":       # manual check:  python3 robot_serial.py /dev/ttyUSB0
    import sys
    r = Robot(sys.argv[1] if len(sys.argv) > 1 else None)
    print("port:", r.port_name, "-> STOP")
    r.stop()
    r.close()
    print("ok")