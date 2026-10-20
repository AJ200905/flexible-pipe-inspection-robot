"""
stream_server.py - tiny MJPEG web server so the laptop can watch the Pi output.

Part 3 helper (shared with Part 2).  Run it through StreamServer:

    from stream_server import StreamServer
    server = StreamServer(5000)
    server.start()
    ...
    server.update(annotated_frame)      # newest frame replaces the old one
    server.info = "6.2 FPS | crack"     # one line of text shown under the video

Then open  http://<pi-hostname>.local:5000  on the laptop.

The server runs in a background thread and never blocks the detection loop:
update() only stores the newest JPEG, the browser pulls it.  If the browser is
slow or closed, frames are simply dropped - detection speed is not affected.
"""
import io
import threading
import time

import cv2

try:
    from flask import Flask, Response, render_template_string
except ImportError:  # keep the detector working even without Flask installed
    Flask = None


PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{ title }}</title>
  <style>
    body   { background:#111; color:#ddd; font-family:sans-serif; margin:0;
             display:flex; flex-direction:column; align-items:center; }
    h1     { font-size:18px; font-weight:600; margin:12px; color:#8f8; }
    img    { max-width:96vw; border:1px solid #333; border-radius:4px; }
    #info  { font-family:monospace; font-size:15px; margin:10px; color:#fc0;
             min-height:20px; }
    .bar   { width:96vw; height:4px; background:#222; margin:8px; }
    .bar i { display:block; height:100%; width:0; background:#4a4; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <img src="/stream.mjpg" alt="stream">
  <div class="bar"><i id="b"></i></div>
  <div id="info">{{ info }}</div>
<script>
  // refresh the text line ~4x a second (image itself is a live stream)
  setInterval(function () {
    fetch('/info.txt').then(r => r.text()).then(t => document.getElementById('info').textContent = t);
    fetch('/fps.txt').then(r => r.text()).then(t => document.getElementById('b').style.width = t + '%');
  }, 250);
</script>
</body>
</html>
"""

BOUNDARY = "part3frame"


class StreamServer:
    """Serve the latest annotated frame as MJPEG on http://host:port/."""

    def __init__(self, port=5000, host="0.0.0.0", title="Pipe Inspection Robot",
                 jpeg_quality=80):
        self.port = int(port)
        self.host = host
        self.title = title
        self.jpeg_quality = int(jpeg_quality)

        self.info = "waiting for the first frame..."
        self.fps_pct = 0.0

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._jpeg = b""
        self._stamp = 0.0
        self._clients = 0
        self._closed = False

        self._app = None
        self._thread = None
        self._server = None            # werkzeug server handle (for close())

    # ---------------------------------------------------------------- frames
    def update(self, frame, fps=None):
        """Hand over a new BGR frame.  Encodes to JPEG and wakes the browser."""
        if self._closed:
            return
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        data = buf.tobytes()
        with self._cond:
            self._jpeg = data
            self._stamp = time.time()
            if fps is not None:
                # 0..6 FPS mapped to 0..100% for the little progress bar
                self.fps_pct = max(0.0, min(100.0, float(fps) / 6.0 * 100.0))
            self._cond.notify_all()

    def close(self):
        """Stop the web server and drop every connected browser."""
        with self._cond:
            self._closed = True
            self._jpeg = b""
            self._stamp = 0.0
            self._cond.notify_all()
        server = self._server
        self._server = None
        if server is not None:
            try:
                server.shutdown()          # stops accepting, wakes the serve loop
            except Exception:
                pass
            try:
                server.server_close()       # release the listening socket
            except Exception:
                pass

    # ---------------------------------------------------------------- routes
    def _stream(self):
        # MJPEG multipart: each part is one JPEG, repeated forever.  We wait for a
        # *new* frame every time (re-sending the same JPEG spins a core), and we
        # keep waiting while the first frame has not arrived yet - a browser that
        # opens the page early must not get an empty stream.
        headers = (f"--{BOUNDARY}\r\n"
                   "Cache-Control: no-cache, no-store, must-revalidate\r\n"
                   "Pragma: no-cache\r\n"
                   "Content-Type: image/jpeg\r\n\r\n").encode()
        last = None
        while True:
            with self._cond:
                if not self._cond.wait_for(
                        lambda: self._closed or self._stamp != last,
                        timeout=2.0):
                    continue                # idle: check again
                if self._closed:
                    return                  # server is finished
                jpeg, last = self._jpeg, self._stamp
            yield headers + jpeg + b"\r\n"

    def _build_app(self):
        if Flask is None:
            raise RuntimeError("Flask is not installed - run: pip install flask")
        app = Flask("pipe_stream")
        app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

        @app.get("/")
        def index():
            return render_template_string(PAGE, title=self.title, info=self.info)

        @app.get("/stream.mjpg")
        def stream():
            return Response(self._stream(),
                            mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                            headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                                     "Pragma": "no-cache",
                                     "Connection": "close"})

        @app.get("/info.txt")
        def info():
            return self.info

        @app.get("/fps.txt")
        def fps_pct():
            return f"{self.fps_pct:.0f}"

        @app.get("/snapshot.jpg")
        def snapshot():
            with self._lock:
                jpeg = self._jpeg
            if not jpeg:
                return ("no frame yet", 503, {"Content-Type": "text/plain"})
            return Response(jpeg, mimetype="image/jpeg",
                            headers={"Cache-Control": "no-store"})

        return app

    # ------------------------------------------------------------ life cycle
    def start(self, block=False):
        """Start serving.  block=True runs in the foreground (Ctrl+C to stop)."""
        self._app = self._build_app()
        from werkzeug.serving import make_server
        self._server = make_server(self.host, self.port, self._app,
                                   threaded=True)
        if block:
            try:
                self._server.serve_forever()
            finally:
                self._server.server_close()
            return
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="stream-server")
        self._thread.start()
        print(f"[stream] open http://<this-pi>:{self.port}/  "
              "(Ctrl+C here stops the detector)")

    def _serve(self):
        # werkzeug's dev server is fine here: one browser, MJPEG, LAN only.
        try:
            self._server.serve_forever()
        except Exception as exc:
            print(f"[stream] server stopped: {exc}")

    def is_up(self):
        return self._server is not None and not self._closed


if __name__ == "__main__":  # manual test: python3 stream_server.py
    srv = StreamServer(port=5000, title="StreamServer test")
    cam = cv2.VideoCapture(0 if len(__import__("sys").argv) < 2 else __import__("sys").argv[1])
    srv.start()
    fps = 0.0
    prev = time.time()
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                frame = 255 - __import__("numpy").zeros((480, 640, 3), "uint8")
                cv2.putText(frame, "no camera", (200, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            srv.update(frame, fps=fps)
            srv.info = f"{fps:.1f} FPS | test pattern"
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        cam.release()