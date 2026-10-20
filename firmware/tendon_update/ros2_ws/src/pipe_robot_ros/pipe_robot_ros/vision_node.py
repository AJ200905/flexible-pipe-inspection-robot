"""
vision_node  -  USB camera + YOLO26n (NCNN) defect detection (runs on the Pi)

Publishes
  /camera/annotated/compressed   sensor_msgs/CompressedImage  frame with boxes + FPS (JPEG)
  /camera/image_raw/compressed   sensor_msgs/CompressedImage  plain frame (only if publish_raw: true)
  /defects                       vision_msgs/Detection2DArray class, score, box (pixels)
  /defects/summary               std_msgs/String              "crack 0.82, hole 0.55"
  /defect_alarm                  std_msgs/String              class name, when a stop_on class is seen

Also saves detections.csv + snapshots in save_dir (same as pi_detect.py).
Set model: "" to run as a plain camera streamer.
"""

import csv
import os
import threading
import time
from datetime import datetime

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

JPEG_FORMAT = "bgr8; jpeg compressed bgr8"   # what image_transport's decoder expects


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")
        p = self.declare_parameter
        self.camera = str(p("camera", "0").value)          # "0", "/dev/video0" or a video file
        self.width = p("width", 640).value
        self.height = p("height", 480).value
        self.model_path = os.path.expanduser(p("model", "~/pipe_robot/best_ncnn_model").value)
        self.imgsz = p("imgsz", 320).value
        self.conf = p("conf", 0.40).value
        self.max_fps = p("max_fps", 10.0).value
        self.quality = p("jpeg_quality", 70).value
        self.publish_raw = p("publish_raw", False).value
        self.stop_on = list(p("stop_on", ["crack"]).value)
        self.save_dir = os.path.expanduser(p("save_dir", "~/pipe_robot/detections").value)
        self.frame_id = p("frame_id", "camera_link").value
        self.loop_video = p("loop_video", True).value

        self.pub_ann = self.create_publisher(CompressedImage, "camera/annotated/compressed", 2)
        self.pub_raw = self.create_publisher(CompressedImage, "camera/image_raw/compressed", 2)
        self.pub_det = self.create_publisher(Detection2DArray, "defects", 10)
        self.pub_sum = self.create_publisher(String, "defects/summary", 10)
        self.pub_alarm = self.create_publisher(String, "defect_alarm", 10)

        # ---- model (optional)
        self.model = None
        if self.model_path:
            from ultralytics import YOLO          # imported here so the camera works without it
            self.get_logger().info(f"Loading model {self.model_path} ...")
            self.model = YOLO(self.model_path, task="detect")
            self.names = self.model.names

        # ---- camera
        self.cap = self.open_camera()
        self.is_file = os.path.isfile(self.camera)

        # ---- logging
        os.makedirs(self.save_dir, exist_ok=True)
        log_path = os.path.join(self.save_dir, "detections.csv")
        new = not os.path.exists(log_path)
        self.log_file = open(log_path, "a", newline="")
        self.log = csv.writer(self.log_file)
        if new:
            self.log.writerow(["time", "class", "confidence", "x1", "y1", "x2", "y2", "image"])
        self.last_snap = 0.0

        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        self.get_logger().info("Vision node running")

    def open_camera(self):
        if self.camera.isdigit():
            cap = cv2.VideoCapture(int(self.camera), cv2.CAP_V4L2)
        elif self.camera.startswith("/dev/video"):
            cap = cv2.VideoCapture(self.camera, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(self.camera)
        if self.camera.isdigit() or self.camera.startswith("/dev/video"):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera '{self.camera}'. Check: v4l2-ctl --list-devices")
        return cap

    def compressed(self, img, stamp):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(self.quality)])
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.format = JPEG_FORMAT
        msg.data = buf.tobytes() if ok else b""
        return msg

    def loop(self):
        fps, t_prev = 0.0, time.time()
        period = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
        while self.running and rclpy.ok():
            t_start = time.time()
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file and self.loop_video:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.get_logger().warn("No frame from camera", throttle_duration_sec=2.0)
                time.sleep(0.2)
                continue

            stamp = self.get_clock().now().to_msg()
            if self.publish_raw:
                self.pub_raw.publish(self.compressed(frame, stamp))

            annotated, found = frame, []
            if self.model is not None:
                r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
                annotated = r.plot()
                found = self.publish_detections(r, stamp, annotated)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            cv2.putText(annotated, f"{fps:.1f} FPS", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            try:
                self.pub_ann.publish(self.compressed(annotated, stamp))
            except Exception:
                if not self.running or not rclpy.ok():
                    break                       # shutting down (Ctrl+C) - exit quietly
                raise

            if period:
                time.sleep(max(0.0, period - (time.time() - t_start)))

    def publish_detections(self, r, stamp, annotated):
        arr = Detection2DArray()
        arr.header.stamp = stamp
        arr.header.frame_id = self.frame_id
        found = []
        for b in r.boxes:
            cls = self.names[int(b.cls)]
            score = float(b.conf)
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            found.append((cls, score, x1, y1, x2, y2))

            d = Detection2D()
            d.header = arr.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = cls
            hyp.hypothesis.score = score
            d.results.append(hyp)
            try:                                   # vision_msgs 4.x (Humble)
                d.bbox.center.position.x = (x1 + x2) / 2
                d.bbox.center.position.y = (y1 + y2) / 2
            except AttributeError:                 # older vision_msgs
                d.bbox.center.x = (x1 + x2) / 2
                d.bbox.center.y = (y1 + y2) / 2
            d.bbox.size_x = x2 - x1
            d.bbox.size_y = y2 - y1
            arr.detections.append(d)
        self.pub_det.publish(arr)

        if found:
            self.pub_sum.publish(String(data=", ".join(f"{c} {s:.2f}" for c, s, *_ in found)))
            hits = sorted({c for c, *_ in found if c in self.stop_on})
            if hits:
                self.pub_alarm.publish(String(data=",".join(hits)))
            self.save(found, annotated)
        return found

    def save(self, found, annotated):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_name = ""
        if time.time() - self.last_snap > 1.0:
            img_name = os.path.join(self.save_dir, f"{stamp}.jpg")
            cv2.imwrite(img_name, annotated)
            self.last_snap = time.time()
        for cls, score, x1, y1, x2, y2 in found:
            self.log.writerow([stamp, cls, f"{score:.2f}", int(x1), int(y1), int(x2), int(y2), img_name])
        self.log_file.flush()

    def destroy_node(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()
        self.log_file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = VisionNode()
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
