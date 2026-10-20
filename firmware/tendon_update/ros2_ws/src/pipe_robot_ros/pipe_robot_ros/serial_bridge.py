"""
serial_bridge  -  ROS 2 <-> ESP32 (runs on the Pi)

Subscribes
  /cmd_vel        geometry_msgs/Twist   linear.x -> all 3 motors (forward/back)
                                        angular.z -> tendon bend (+ left / - right)
  /bend           std_msgs/Int32        hold a bend: -100 (full right) .. 0 .. +100 (full left)
  /light          std_msgs/Int32        0..100 %
  /servo          std_msgs/Int32        raw servo angle 0..180 deg (calibration only)
  /zero_imu       std_msgs/Empty        current pose = level
  /defect_alarm   std_msgs/String       from vision_node -> stop and hold (if enabled)

Publishes
  /imu            sensor_msgs/Imu       orientation from roll/pitch (yaw = 0)
  /motor_output   std_msgs/Float32MultiArray  [cmd1, cmd2, cmd3, rpm1, rpm2, rpm3]
  /robot_status   std_msgs/String       "roll=.. pitch=.. light=.. servo=.. link=ok"
  /robot_marker   visualization_msgs/Marker   box that tilts with the robot in RViz
  TF  world -> base_link  (rotation only)
"""

import math
import time

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty, Float32MultiArray, Int32, String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker

from .esp32_link import Esp32Link


def quat_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,   # x
            cr * sp * cy + sr * cp * sy,   # y
            cr * cp * sy - sr * sp * cy,   # z
            cr * cp * cy + sr * sp * sy)   # w


class SerialBridge(Node):
    def __init__(self):
        super().__init__("serial_bridge")
        p = self.declare_parameter
        self.port = p("port", "auto").value
        self.baud = p("baud", 115200).value
        self.max_linear = p("max_linear_speed", 1.0).value      # m/s that maps to 100 %
        self.steer_gain = p("steer_gain_pct", 100.0).value       # bend % per rad/s of angular.z
        self.max_pct = p("max_pct", 80.0).value                  # safety cap
        self.cmd_timeout = p("cmd_timeout", 0.5).value           # s without /cmd_vel -> stop
        self.stop_on_defect = p("stop_on_defect", False).value
        self.defect_hold = p("defect_hold_s", 3.0).value
        self.frame_world = p("world_frame", "world").value
        self.frame_base = p("base_frame", "base_link").value
        self.imu_frame = p("imu_frame", "base_link").value

        self.link = None
        self.last_cmd = 0.0              # motor % (same for all 3)
        self.last_turn = 0.0             # angular.z from teleop
        self.bend_hold = 0               # from /bend topic
        self.sent_bend = None
        self.last_bend_send = 0.0
        self.raw_servo = False           # True after /servo: leave the raw angle alone
        self.last_cmd_time = 0.0
        self.moving = False
        self.hold_until = 0.0
        self.last_tel_stamp = 0.0
        self.last_connect_try = 0.0

        self.create_subscription(Twist, "cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(Int32, "light", lambda m: self.send(f"L {int(m.data)}"), 10)
        self.create_subscription(Int32, "servo", self.on_servo_raw, 10)
        self.create_subscription(Int32, "bend", self.on_bend, 10)
        self.create_subscription(Empty, "zero_imu", lambda m: self.send("Z"), 10)
        self.create_subscription(String, "defect_alarm", self.on_defect, 10)

        self.pub_imu = self.create_publisher(Imu, "imu", 10)
        self.pub_motor = self.create_publisher(Float32MultiArray, "motor_output", 10)
        self.pub_status = self.create_publisher(String, "robot_status", 10)
        self.pub_marker = self.create_publisher(Marker, "robot_marker", 1)
        self.tf = TransformBroadcaster(self)

        self.connect()
        self.create_timer(0.1, self.control_tick)    # 10 Hz: motor commands + heartbeat
        self.create_timer(0.05, self.telemetry_tick)  # 20 Hz: publish sensor data
        self.create_timer(0.5, self.marker_tick)

    # ---------------------------------------------------------- connection
    def connect(self):
        self.last_connect_try = time.time()
        try:
            self.link = Esp32Link(self.port, self.baud, on_message=self.on_esp_message)
            self.get_logger().info(f"ESP32 connected on {self.link.port}")
        except Exception as e:
            self.link = None
            self.get_logger().warn(f"ESP32 not connected ({e}). Retrying every 3 s ...")

    def on_esp_message(self, line):
        self.get_logger().info(f"ESP32: {line}")

    def send(self, text):
        if self.link and self.link.alive:
            self.link.send(text)

    # ---------------------------------------------------------- inputs
    def on_cmd_vel(self, msg):
        v = max(-1.0, min(1.0, msg.linear.x / self.max_linear)) * 100.0
        self.last_cmd = max(-self.max_pct, min(self.max_pct, v))
        self.last_turn = msg.angular.z
        self.last_cmd_time = time.time()

    def on_bend(self, msg):
        self.bend_hold = max(-100, min(100, int(msg.data)))
        self.raw_servo = False
        self.sent_bend = None                       # force re-send

    def on_servo_raw(self, msg):
        self.bend_hold = 0
        self.raw_servo = True                       # don't override the raw angle
        self.send(f"V {int(msg.data)}")

    def on_defect(self, msg):
        if self.stop_on_defect:
            self.hold_until = time.time() + self.defect_hold
            self.send("S")
            self.get_logger().warn(f"Defect '{msg.data}' -> stopping for {self.defect_hold:.0f} s")

    # ---------------------------------------------------------- timers
    def control_tick(self):
        if self.link is None or not self.link.alive:
            if time.time() - self.last_connect_try > 3.0:
                self.connect()
            return

        fresh = time.time() - self.last_cmd_time < self.cmd_timeout
        v = self.last_cmd if fresh else 0.0
        turn = self.last_turn if fresh else 0.0
        if time.time() < self.hold_until:          # defect hold: no forward motion
            v = min(v, 0.0)

        # tendon bend = held bend (/bend) + live steering from teleop
        bend = int(max(-100, min(100, self.bend_hold + turn * self.steer_gain)))
        if self.raw_servo and abs(turn) > 0.01:     # steering again -> leave calibration mode
            self.raw_servo = False
            self.sent_bend = None
        if not self.raw_servo and (bend != self.sent_bend or time.time() - self.last_bend_send > 1.0):
            self.send(f"D {bend}")
            self.sent_bend = bend
            self.last_bend_send = time.time()

        if abs(v) > 0.5:
            self.send(f"M {v:.0f} {v:.0f} {v:.0f}")
            self.moving = True
        else:
            if self.moving:
                self.send("S")
                self.moving = False
            self.send("P")                          # heartbeat keeps the ESP32 watchdog happy

    def telemetry_tick(self):
        if self.link is None or self.link.telemetry is None:
            return
        if self.link.telemetry_time == self.last_tel_stamp:
            return
        self.last_tel_stamp = self.link.telemetry_time
        t = self.link.telemetry
        now = self.get_clock().now().to_msg()
        qx, qy, qz, qw = quat_from_rpy(math.radians(t["roll"]), math.radians(t["pitch"]), 0.0)

        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = self.imu_frame
        imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w = qx, qy, qz, qw
        imu.orientation_covariance = [0.0003, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0, 0.0, 1e6]
        imu.angular_velocity_covariance[0] = -1.0       # not provided
        imu.linear_acceleration_covariance[0] = -1.0    # not provided
        self.pub_imu.publish(imu)

        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = self.frame_world
        tf.child_frame_id = self.frame_base
        tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tf.sendTransform(tf)

        self.pub_motor.publish(Float32MultiArray(data=[
            t["cmd1"], t["cmd2"], t["cmd3"], t["rpm1"], t["rpm2"], t["rpm3"]]))
        self.pub_status.publish(String(data=(
            f"roll={t['roll']:.1f} pitch={t['pitch']:.1f} "
            f"motors={t['cmd1']:.0f},{t['cmd2']:.0f},{t['cmd3']:.0f} "
            f"light={t['light']:.0f} servo={t['servo']:.0f} bend={self.sent_bend} link=ok")))

    def marker_tick(self):
        m = Marker()
        m.header.frame_id = self.frame_base
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id = "robot", 0
        m.type, m.action = Marker.CUBE, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = 0.20, 0.12, 0.08     # robot body (m) - edit to match yours
        m.color.r, m.color.g, m.color.b, m.color.a = 0.16, 0.47, 0.84, 0.9
        m.frame_locked = True
        self.pub_marker.publish(m)

    def destroy_node(self):
        if self.link:
            self.link.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = SerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
