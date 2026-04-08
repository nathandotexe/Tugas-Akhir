import os, cv2, time, threading, queue, subprocess, signal, socket
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
import tkinter as tk

from flask import Flask, Response

MODEL_PATH        = "Yolov11/best.pt"
H_PATH            = "H.npy"
GRAPH_OUTPUT_TIME = "speed_graph_time.png"
GRAPH_OUTPUT_DIST = "speed_graph_dist.png"
VIDEO_OUTPUT      = "output_tracked.mp4"
DISPLAY_SIZE      = (1280, 720)
CLASSES_TO_TRACK  = [0]
MEDIAN_WIN        = 7
LOG_EVERY_N       = 1
SKIP_FRAMES       = 1
CONF_THRESHOLD    = 0.35
IOU_THRESHOLD     = 0.45

CALIB_PATH        = "calib.npz"
RTSP_URL          = "rtsp://localhost:8554/live"
FLASK_PORT        = 5000
FLASK_ENABLED     = True

MEDIAMTX_EXE     = "mediamtx.exe"
MEDIAMTX_YML     = "mediamtx.yml"
RTMP_PORT         = 1935
RTSP_PORT         = 8554

CORNER_LABELS = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
EDGE_NAMES    = ["Top (TL->TR)", "Right (TR->BR)", "Bottom (BR->BL)", "Left (BL->TL)"]
EDGES         = [(0,1),(1,2),(2,3),(3,0)]
COLORS        = [(0,255,255),(0,200,255),(0,150,255),(0,100,255)]

class MediaMTXManager:
    def __init__(self):
        self._proc = None

    def _kill_port(self, port):
        """Kill any process currently occupying a TCP port."""
        try:
            out = subprocess.check_output(
                f'netstat -ano | findstr :{port}', shell=True, text=True
            )
            pids = set()
            for line in out.splitlines():
                parts = line.split()
                if parts and parts[-1].isdigit():
                    pids.add(parts[-1])
            for pid in pids:
                subprocess.call(f'taskkill /PID {pid} /F', shell=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if pids:
                print(f"  [MediaMTX] Cleared {len(pids)} process(es) from port {port}")
                time.sleep(0.5)
        except Exception:
            pass

    def _port_ready(self, port, timeout=10.0):
        """Wait until a TCP port is accepting connections."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    return True
            except OSError:
                time.sleep(0.3)
        return False

    def start(self):
        if not os.path.exists(MEDIAMTX_EXE):
            print(f"  [MediaMTX] EXE not found: {MEDIAMTX_EXE}")
            print("  [MediaMTX] Set MEDIAMTX_EXE at the top of the script to the correct path.")
            return False

        print("  [MediaMTX] Clearing ports 1935 and 8554...")
        self._kill_port(RTMP_PORT)
        self._kill_port(RTSP_PORT)

        print("  [MediaMTX] Starting...")
        self._proc = subprocess.Popen(
            [MEDIAMTX_EXE, MEDIAMTX_YML],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

        if self._port_ready(RTMP_PORT):
            print(f"  [MediaMTX] Ready — RTMP :1935  RTSP :8554")
            return True
        else:
            print("  [MediaMTX] Timed out waiting for port 1935. Check MEDIAMTX_EXE path.")
            self.stop()
            return False

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            print("  [MediaMTX] Stopped.")
        self._proc = None


def load_calibration(calib_path):
    if calib_path is None or not os.path.exists(calib_path):
        print(f"  [Calib] Not found: {calib_path} — skipping undistortion.")
        return None, None

    data = np.load(calib_path)

    if "map1" in data and "map2" in data:
        print(f"  [Calib] Loaded undistort maps from {calib_path}")
        return data["map1"], data["map2"]

    if "camera_matrix" in data and "dist_coeffs" in data:
        K    = data["camera_matrix"]
        dist = data["dist_coeffs"]
        w, h = data["image_size"].astype(int)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_16SC2)
        print(f"  [Calib] Computed undistort maps from {calib_path}")
        return map1, map2

    print("  [Calib] Unrecognised .npz format — skipping undistortion.")
    return None, None


def undistort_frame(frame, map1, map2):
    if map1 is None or map2 is None:
        return frame
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)



class FlaskStreamer:
    def __init__(self, port):
        self._port  = port
        self._frame = None
        self._lock  = threading.Lock()
        self._app   = Flask(__name__)
        self._app.add_url_rule("/",       "index",  self._index)
        self._app.add_url_rule("/stream", "stream", self._stream)
        threading.Thread(target=self._serve, daemon=True).start()
        print(f"  [Flask] Live stream -> http://0.0.0.0:{self._port}")

    def push(self, frame):
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self._lock:
            self._frame = buf.tobytes()

    def _generate(self):
        while True:
            with self._lock:
                data = self._frame
            if data:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            time.sleep(0.033)

    def _stream(self):
        return Response(self._generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def _index(self):
        return ("<html><head><title>Athlete Speed Tracker</title>"
                "<style>body{background:#111;margin:0;display:flex;"
                "justify-content:center;align-items:center;height:100vh;}"
                "img{max-width:100%;border:2px solid #0f0;}</style></head>"
                "<body><img src='/stream'></body></html>")

    def _serve(self):
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        self._app.run(host="0.0.0.0", port=self._port, threaded=True)


def open_capture(source):
    is_rtsp = isinstance(source, str) and source.startswith("rtsp://")
    if is_rtsp:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"  [RTSP] Stream opened: {source}")
            return cap
        raise RuntimeError(f"Cannot open RTSP stream: {source}")
    backends = [cv2.CAP_MSMF, cv2.CAP_FFMPEG, cv2.CAP_DSHOW, cv2.CAP_ANY]
    for attempt in range(1, 4):
        for backend in backends:
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                print(f"  Capture opened (attempt {attempt}, backend {backend}): {source}")
                return cap
            cap.release()
        print(f"  Attempt {attempt} failed, retrying...")
        time.sleep(0.4)
    raise RuntimeError(f"Cannot open video source: '{source}'")


def grab_first_frame(source, map1, map2):
    cap = open_capture(source)
    frame = None
    for _ in range(10):
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
            break
    cap.release()
    if frame is None:
        raise RuntimeError("Could not decode any frame.")
    frame = undistort_frame(frame, map1, map2)
    return cv2.resize(frame, DISPLAY_SIZE)


def gui_select_mode():
    result = {"mode": None, "path": None}
    root = tk.Tk()
    root.title("Athlete Speed Tracker")
    root.geometry("340x220")
    root.resizable(False, False)

    def set_mode(m):
        result["mode"] = m
        root.destroy()

    tk.Label(root, text="Athlete Speed Tracker", font=("Arial", 15, "bold")).pack(pady=12)
    tk.Label(root, text="Select input source:").pack()
    tk.Button(root, text="Offline Video File", width=26,
              command=lambda: set_mode("offline")).pack(pady=6)
    tk.Button(root, text="Real-Time Camera", width=26,
              command=lambda: set_mode("realtime")).pack(pady=6)
    root.mainloop()

    if result["mode"] == "offline":
        root2 = tk.Tk()
        root2.withdraw()
        path = filedialog.askopenfilename(
            parent=root2, title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4"), ("All files", "*.*")]
        )
        root2.destroy()
        if not path:
            return None, None
        result["path"] = path

    source = result["path"] if result["mode"] == "offline" else RTSP_URL
    return result["mode"], source


def gui_pick_corners(frame):
    pts   = []
    clone = frame.copy()
    WIN   = "FIELD CORNERS: TL > TR > BR > BL  |  BACKSPACE=undo  ENTER=confirm  ESC=cancel"

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            idx = len(pts) - 1
            cv2.circle(clone, (x, y), 8, COLORS[idx], -1)
            cv2.putText(clone, CORNER_LABELS[idx], (x+10, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[idx], 2)
            if len(pts) > 1:
                cv2.line(clone, pts[-2], pts[-1], (0,255,180), 2)
            if len(pts) == 4:
                cv2.line(clone, pts[-1], pts[0], (0,255,180), 2)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, *DISPLAY_SIZE)
    disp = clone.copy()
    cv2.putText(disp, f"Next: {CORNER_LABELS[0]}  |  0/4  |  ENTER=confirm  ESC=cancel",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.imshow(WIN, disp)
    cv2.waitKey(1)
    cv2.setMouseCallback(WIN, on_click)

    while True:
        disp = clone.copy()
        label = CORNER_LABELS[len(pts)] if len(pts) < 4 else "All 4 -- press ENTER"
        cv2.putText(disp, f"Next: {label}  |  {len(pts)}/4  |  ENTER=confirm  ESC=cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow(WIN, disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, ord('\r')) and len(pts) == 4:
            break
        if key == 27:
            pts.clear()
            break
        if key == 8 and pts:
            pts.pop()
            clone = frame.copy()
            for i, p in enumerate(pts):
                cv2.circle(clone, p, 8, COLORS[i], -1)
                cv2.putText(clone, CORNER_LABELS[i], (p[0]+10, p[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[i], 2)
            for i in range(1, len(pts)):
                cv2.line(clone, pts[i-1], pts[i], (0,255,180), 2)

    cv2.destroyWindow(WIN)
    return pts if len(pts) == 4 else None


def gui_pick_start_finish():
    result = {"val": None}
    root = tk.Tk()
    root.title("Choose Start / Finish Edges & Distance")
    root.geometry("400x260")
    root.resizable(False, False)

    tk.Label(root, text="Choose Start & Finish Lines",
             font=("Arial", 13, "bold")).pack(pady=10)
    tk.Label(root, text="Corners were set as: TL -> TR -> BR -> BL").pack()

    frm = tk.Frame(root); frm.pack(pady=8)
    tk.Label(frm, text="START edge:").grid(row=0, column=0, padx=8, sticky="e")
    start_var = tk.StringVar(value=EDGE_NAMES[3])
    ttk.Combobox(frm, textvariable=start_var, values=EDGE_NAMES,
                 state="readonly", width=22).grid(row=0, column=1)

    tk.Label(frm, text="FINISH edge:").grid(row=1, column=0, padx=8, pady=6, sticky="e")
    finish_var = tk.StringVar(value=EDGE_NAMES[1])
    ttk.Combobox(frm, textvariable=finish_var, values=EDGE_NAMES,
                 state="readonly", width=22).grid(row=1, column=1)

    tk.Label(frm, text="Distance (meters):").grid(row=2, column=0, padx=8, pady=6, sticky="e")
    dist_var = tk.StringVar(value="10.0")
    tk.Entry(frm, textvariable=dist_var, width=24).grid(row=2, column=1)

    def confirm():
        si = EDGE_NAMES.index(start_var.get())
        fi = EDGE_NAMES.index(finish_var.get())
        if si == fi:
            messagebox.showwarning("Invalid", "Start and Finish must be different.")
            return
        try:
            dist_val = float(dist_var.get())
            if dist_val <= 0: raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid", "Distance must be a positive number.")
            return
        result["val"] = (si, fi, dist_val)
        root.destroy()

    tk.Button(root, text="Confirm", width=14, command=confirm).pack(side="left", padx=40, pady=8)
    tk.Button(root, text="Cancel",  width=14, command=root.destroy).pack(side="right", padx=40, pady=8)
    root.mainloop()
    return result["val"]


def show_field_preview(frame, corners):
    preview = frame.copy()
    cv2.polylines(preview, [np.array(corners, dtype=np.int32)],
                  isClosed=True, color=(0,255,180), thickness=2)
    for i, (ei, ej) in enumerate(EDGES):
        mx = (corners[ei][0] + corners[ej][0]) // 2
        my = (corners[ei][1] + corners[ej][1]) // 2
        cv2.putText(preview, EDGE_NAMES[i], (mx-40, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
    for i, c in enumerate(corners):
        cv2.circle(preview, c, 6, COLORS[i], -1)
        cv2.putText(preview, CORNER_LABELS[i], (c[0]+8, c[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[i], 1)
    WIN = "Field preview -- press any key to continue"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, *DISPLAY_SIZE)
    cv2.imshow(WIN, preview)
    cv2.waitKey(0)
    cv2.destroyWindow(WIN)


def run_field_setup(frame, H):
    corners = gui_pick_corners(frame)
    if corners is None:
        return None
    show_field_preview(frame, corners)
    sf = gui_pick_start_finish()
    if sf is None:
        return None
    start_idx, finish_idx, distance_m = sf
    i0, i1 = EDGES[start_idx]
    j0, j1 = EDGES[finish_idx]
    start_seg  = (corners[i0], corners[i1])
    finish_seg = (corners[j0], corners[j1])
    mask = np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0]), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(corners, dtype=np.int32)], 255)
    return corners, start_seg, finish_seg, mask, distance_m


def load_homography(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Homography file '{path}' not found.")
    return np.load(path)


def pixel_to_world(H, px, py):
    pt = np.array([[[px, py]]], dtype=np.float32)
    w  = cv2.perspectiveTransform(pt, H)
    return float(w[0,0,0]), float(w[0,0,1])


def world_to_pixel(H_inv, wx, wy):
    pt = np.array([[[wx, wy]]], dtype=np.float32)
    p  = cv2.perspectiveTransform(pt, H_inv)
    return float(p[0,0,0]), float(p[0,0,1])


def segments_cross(p1, p2, q1, q2):
    def cross2d(a, b): return a[0]*b[1] - a[1]*b[0]
    d1 = (p2[0]-p1[0], p2[1]-p1[1])
    d2 = (q2[0]-q1[0], q2[1]-q1[1])
    denom = cross2d(d1, d2)
    if abs(denom) < 1e-9: return False
    t = cross2d((q1[0]-p1[0], q1[1]-p1[1]), d2) / denom
    u = cross2d((q1[0]-p1[0], q1[1]-p1[1]), d1) / denom
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def point_in_polygon(px, py, corners):
    poly = np.array(corners, dtype=np.float32)
    return cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0


def make_kalman(fps, wx, wy):
    dt = 1.0 / fps
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
    kf.H = np.eye(2, 4, dtype=float)
    kf.x = np.array([[wx],[wy],[0.0],[0.0]])
    kf.P = np.diag([5.0, 5.0, 2.0, 2.0])
    kf.R = np.diag([0.3, 0.3])
    kf.Q = np.diag([0.05, 0.05, 0.5, 0.5])
    return kf


class AsyncWriter:
    def __init__(self, path, fps, size):
        self._writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self._q = queue.Queue(maxsize=64)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def write(self, frame):
        try: self._q.put_nowait(frame.copy())
        except queue.Full: pass

    def _run(self):
        while True:
            f = self._q.get()
            if f is None: break
            self._writer.write(f)

    def release(self):
        self._q.put(None)
        self._t.join()
        self._writer.release()


class TrackState:
    __slots__ = ("kf","prev_t","prev_px","speed_buf","last_speed_ms",
                 "color","start_crossed","finish_crossed","total_dist",
                 "last_seen_t","last_box","last_disp_kh")

    def __init__(self, kf, t, px, py, color, box):
        self.kf             = kf
        self.prev_t         = t
        self.prev_px        = (px, py)
        self.speed_buf      = deque(maxlen=MEDIAN_WIN)
        self.last_speed_ms  = 0.0
        self.color          = color
        self.start_crossed  = []
        self.finish_crossed = []
        self.total_dist     = 0.0
        self.last_seen_t    = t
        self.last_box       = box
        self.last_disp_kh   = 0.0


def generate_speed_graphs(speed_logs, start_events, finish_events, out_path_time, out_path_dist):
    if not speed_logs:
        print("No speed data."); return
    from scipy.ndimage import gaussian_filter1d
    n = len(speed_logs)
    palette = plt.cm.tab10.colors

    for out_path, x_key, xlabel, title_suffix in [
        (out_path_time, "time", "Time (s)",     "Over Time"),
        (out_path_dist, "dist", "Distance (m)", "Over Distance"),
    ]:
        fig, axes = plt.subplots(n, 1, figsize=(13, 3.8*n), squeeze=False)
        fig.suptitle(f"Athlete Speed Profiles ({title_suffix})",
                     fontsize=16, fontweight="bold", y=1.01)

        for row, (tid, records) in zip(axes, sorted(speed_logs.items())):
            ax = row[0]
            if len(records) < 3:
                ax.set_title(f"ID {tid} -- insufficient data"); continue
            ts   = np.array([r[0] for r in records])
            kmph = np.array([r[1] for r in records]) * 3.6
            ds   = np.array([r[2] if len(r) > 2 else 0.0 for r in records])
            xs   = ts if x_key == "time" else ds
            col  = palette[tid % len(palette)]
            ax.plot(xs, kmph, color=col, alpha=0.2, linewidth=1)
            smooth = gaussian_filter1d(kmph, sigma=4)
            ax.plot(xs, smooth, color=col, linewidth=2.3, label=f"ID {tid} (km/h)")

            for st in start_events.get(tid, []):
                xv = st if x_key == "time" else ds[np.searchsorted(ts, st)] if np.searchsorted(ts, st) < len(ds) else ds[-1]
                ax.axvline(xv, color="lime", linestyle=":", linewidth=1.5, label="Start")
            for ft in finish_events.get(tid, []):
                xv = ft if x_key == "time" else ds[np.searchsorted(ts, ft)] if np.searchsorted(ts, ft) < len(ds) else ds[-1]
                ax.axvline(xv, color="red", linestyle="--", linewidth=1.5, label="Finish")

            pk = int(np.argmax(smooth))
            ax.annotate(f"Peak {smooth[pk]:.1f} km/h", xy=(xs[pk], smooth[pk]),
                        xytext=(xs[pk]+0.5, smooth[pk]+1.5), fontsize=8, color="darkred",
                        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2))
            ax.set_ylabel("Speed (km/h)"); ax.set_xlabel(xlabel)
            ax.set_title(f"ID {tid}  Peak:{max(kmph):.1f}  Avg:{np.mean(kmph):.1f} km/h")
            ax.set_ylim(bottom=0); ax.grid(True, alpha=0.3)
            h, l = ax.get_legend_handles_labels()
            seen, uh, ul = set(), [], []
            for hh, ll in zip(h, l):
                if ll not in seen: seen.add(ll); uh.append(hh); ul.append(ll)
            ax.legend(uh, ul, fontsize=8, loc="upper right")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Graph saved -> {out_path}")


def main():
    mode, source = gui_select_mode()
    if mode is None:
        print("Cancelled."); return
    print(f"\nMode: {mode}  |  Source: {source}")

    mediamtx = None
    if mode == "realtime":
        mediamtx = MediaMTXManager()
        if not mediamtx.start():
            print("  [MediaMTX] Failed to start — continuing anyway (manual start may work).")

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available(): device = "cuda"
        elif torch.backends.mps.is_available(): device = "mps"
    except Exception: pass
    use_half = device in ("cuda", "mps")
    print(f"Device: {device.upper()}  fp16: {use_half}")

    print("Loading calibration...")
    calib_map1, calib_map2 = load_calibration(CALIB_PATH)

    flask_streamer = FlaskStreamer(FLASK_PORT) if FLASK_ENABLED else None

    print(f"Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    H = load_homography(H_PATH)
    H_inv = np.linalg.inv(H)

    if mode == "realtime":
        print("\n  Waiting for DJI stream...")
        print("  -> Open DJI Fly, go to Transmission -> Live Streaming -> RTMP")
        print(f"  -> Enter: rtmp://YOUR_PC_IP:1935/live  then tap Start")
        print("  -> Once the red live icon appears in DJI Fly, press Enter here to continue.")
        input("  Press Enter when drone is streaming...\n")

    print("Grabbing first frame...")
    first_frame = grab_first_frame(source, calib_map1, calib_map2)

    setup = run_field_setup(first_frame, H)
    if setup is None:
        print("Field setup cancelled.")
        if mediamtx: mediamtx.stop()
        return
    corners, start_seg, finish_seg, roi_mask, distance_m = setup

    scx = (start_seg[0][0] + start_seg[1][0]) / 2.0
    scy = (start_seg[0][1] + start_seg[1][1]) / 2.0
    fcx = (finish_seg[0][0] + finish_seg[1][0]) / 2.0
    fcy = (finish_seg[0][1] + finish_seg[1][1]) / 2.0
    ws_x, ws_y = pixel_to_world(H, scx, scy)
    wf_x, wf_y = pixel_to_world(H, fcx, fcy)
    world_dist = np.hypot(wf_x - ws_x, wf_y - ws_y)
    scale_m_per_w = distance_m / world_dist if world_dist > 0 and distance_m > 0 else 1.0
    print(f"  [Geometry] {distance_m}m = {world_dist:.2f} world units  (scale {scale_m_per_w:.4f} m/wu)")

    print("Opening capture...")
    cap     = open_capture(source)
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    if not vid_fps or vid_fps <= 0:
        vid_fps = 30.0
        print("  [FPS] Stream did not report FPS -- defaulting to 30fps")
    print(f"FPS: {vid_fps:.2f}")

    writer = AsyncWriter(VIDEO_OUTPUT, vid_fps, DISPLAY_SIZE) if VIDEO_OUTPUT else None

    tracks        = {}
    speed_logs    = defaultdict(list)
    start_events  = defaultdict(list)
    finish_events = defaultdict(list)

    rng = np.random.default_rng(42)
    def rand_color(_): return tuple(int(v) for v in rng.integers(60, 230, 3))

    frame_idx       = 0
    fps_display     = 0.0
    t_fps           = time.perf_counter()
    last_det_result = None
    paused          = False

    print("\nTracking -- ESC=stop" + ("  R=recalibrate" if mode == "realtime" else "") + "\n")

    while True:
        if paused:
            ret, live = cap.read()
            if ret and live is not None:
                live = undistort_frame(live, calib_map1, calib_map2)
                live = cv2.resize(live, DISPLAY_SIZE)
                new_setup = run_field_setup(live, H)
                if new_setup is not None:
                    corners, start_seg, finish_seg, roi_mask, distance_m = new_setup
                    scx = (start_seg[0][0] + start_seg[1][0]) / 2.0
                    scy = (start_seg[0][1] + start_seg[1][1]) / 2.0
                    fcx = (finish_seg[0][0] + finish_seg[1][0]) / 2.0
                    fcy = (finish_seg[0][1] + finish_seg[1][1]) / 2.0
                    ws_x, ws_y = pixel_to_world(H, scx, scy)
                    wf_x, wf_y = pixel_to_world(H, fcx, fcy)
                    world_dist = np.hypot(wf_x - ws_x, wf_y - ws_y)
                    scale_m_per_w = distance_m / world_dist if world_dist > 0 else 1.0
                    print(f"  [Geometry] Recalibrated: {distance_m}m = {world_dist:.2f} wu (scale {scale_m_per_w:.4f})")
                    tracks.clear()
            paused = False

        ret, frame = cap.read()
        if not ret:
            print("End of stream."); break

        frame = undistort_frame(frame, calib_map1, calib_map2)
        frame = cv2.resize(frame, DISPLAY_SIZE)
        frame_idx += 1
        t_now = (frame_idx / vid_fps) if mode == "offline" else time.perf_counter()

        masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

        if (frame_idx % SKIP_FRAMES == 0) or frame_idx == 1:
            results = model.track(
                masked_frame, persist=True, tracker="bytetrack.yaml",
                classes=CLASSES_TO_TRACK, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                half=use_half, device=device, verbose=False, imgsz=640,
            )
            last_det_result = results[0]

        if last_det_result is None:
            cv2.imshow("Speed Tracker", frame)
            if cv2.waitKey(1) == 27: break
            continue

        annotated   = frame.copy()
        current_ids = set()

        det = last_det_result
        if det.boxes is not None and det.boxes.id is not None:
            boxes = det.boxes.xyxy.cpu().numpy()
            ids   = det.boxes.id.cpu().numpy().astype(int)
            confs = det.boxes.conf.cpu().numpy()

            for box, tid, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box
                cx_px = (x1 + x2) * 0.5
                cy_px = float(y2)

                if not point_in_polygon(int(cx_px), int(cy_px), corners):
                    continue

                current_ids.add(tid)
                wx, wy   = pixel_to_world(H, cx_px, cy_px)
                wx_m     = wx * scale_m_per_w
                wy_m     = wy * scale_m_per_w

                if tid not in tracks:
                    tracks[tid] = TrackState(
                        kf=make_kalman(vid_fps, wx_m, wy_m), t=t_now,
                        px=int(cx_px), py=int(cy_px),
                        color=rand_color(tid), box=(x1,y1,x2,y2)
                    )

                state   = tracks[tid]
                kf      = state.kf
                q_scale = 1.0 + (1.0 - float(conf)) * 2.0
                kf.Q    = np.diag([0.05, 0.05, 0.5*q_scale, 0.5*q_scale])
                kf.predict()
                kf.update([wx_m, wy_m])

                speed_ms = np.hypot(float(kf.x[2].item()), float(kf.x[3].item()))
                state.speed_buf.append(speed_ms)
                disp_ms = float(np.median(state.speed_buf))
                disp_kh = disp_ms * 3.6

                if state.prev_t > 0:
                    state.total_dist += disp_ms * (t_now - state.prev_t)

                state.last_speed_ms = speed_ms
                state.prev_t        = t_now
                state.last_seen_t   = t_now
                state.last_box      = (x1, y1, x2, y2)
                state.last_disp_kh  = disp_kh

                new_px, new_py = int(cx_px), int(cy_px)
                ppx, ppy = state.prev_px

                if segments_cross((ppx,ppy),(new_px,new_py), start_seg[0], start_seg[1]):
                    state.start_crossed.append(t_now)
                    start_events[tid].append(t_now)
                    print(f"  START  ID {tid}  t={t_now:.2f}s  {disp_kh:.1f} km/h")

                if segments_cross((ppx,ppy),(new_px,new_py), finish_seg[0], finish_seg[1]):
                    state.finish_crossed.append(t_now)
                    finish_events[tid].append(t_now)
                    print(f"  FINISH ID {tid}  t={t_now:.2f}s  {disp_kh:.1f} km/h")

                state.prev_px = (new_px, new_py)

                if frame_idx % LOG_EVERY_N == 0:
                    speed_logs[tid].append((t_now, disp_ms, state.total_dist))

        for tid, state in list(tracks.items()):
            if t_now - state.last_seen_t > 2.0:
                del tracks[tid]
                continue

            col     = state.color
            disp_kh = state.last_disp_kh

            if tid not in current_ids:
                dt = t_now - state.prev_t
                if dt > 0:
                    state.kf.predict()
                    disp_ms = float(np.median(state.speed_buf)) if state.speed_buf else 0.0
                    state.total_dist += disp_ms * dt
                    state.prev_t = t_now

                pred_wx_m = float(state.kf.x[0].item())
                pred_wy_m = float(state.kf.x[1].item())
                new_cx, new_cy = world_to_pixel(H_inv, pred_wx_m / scale_m_per_w, pred_wy_m / scale_m_per_w)
                old_x1, old_y1, old_x2, old_y2 = state.last_box
                old_cx, old_cy = (old_x1 + old_x2) * 0.5, float(old_y2)
                dx, dy = new_cx - old_cx, new_cy - old_cy
                x1 = old_x1 + dx; y1 = old_y1 + dy
                x2 = old_x2 + dx; y2 = old_y2 + dy
                state.last_box = (x1, y1, x2, y2)
                col = tuple(max(0, c - 60) for c in col)
                cv2.rectangle(annotated, (int(x1),int(y1)), (int(x2),int(y2)), col, 2, cv2.LINE_4)
                if frame_idx % LOG_EVERY_N == 0:
                    disp_ms = float(np.median(state.speed_buf)) if state.speed_buf else 0.0
                    speed_logs[tid].append((t_now, disp_ms, state.total_dist))
            else:
                x1, y1, x2, y2 = state.last_box
                cv2.rectangle(annotated, (int(x1),int(y1)), (int(x2),int(y2)), col, 2)

            label = f"ID {tid}  {disp_kh:.1f}km/h  {state.total_dist:.1f}m"
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            lx = max(int(x1), 0)
            ly = max(int(y1)-8, th+4)
            cv2.rectangle(annotated, (lx,ly-th-4), (lx+tw+6,ly+2), col, -1)
            cv2.putText(annotated, label, (lx+3,ly-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2, cv2.LINE_AA)

            if state.start_crossed and not state.finish_crossed:
                elapsed = t_now - state.start_crossed[-1]
                cv2.putText(annotated, f"SPRINT {elapsed:.1f}s",
                            (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0,255,255), 2, cv2.LINE_AA)
            elif state.finish_crossed:
                cv2.putText(annotated, "FINISH",
                            (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0,80,255), 2, cv2.LINE_AA)

        t_wall      = time.perf_counter()
        fps_display = 0.9*fps_display + 0.1/max(t_wall-t_fps, 1e-9)
        t_fps       = t_wall
        hint        = "ESC=stop  R=recalibrate" if mode == "realtime" else "ESC=stop"
        cv2.putText(annotated,
                    f"FPS:{fps_display:.1f}  Frame:{frame_idx}  [{device.upper()}]  {hint}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,255,0), 2, cv2.LINE_AA)

        if flask_streamer: flask_streamer.push(annotated)
        cv2.imshow("Speed Tracker", annotated)
        if writer: writer.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        if key in (ord('r'), ord('R')) and mode == "realtime":
            print("  R -- recalibrating...")
            paused = True

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    if mediamtx: mediamtx.stop()

    print("\n--- Summary ---")
    for tid in sorted(speed_logs):
        spds = [r[1] for r in speed_logs[tid]]
        if not spds: continue
        dist  = speed_logs[tid][-1][2] if len(speed_logs[tid][-1]) > 2 else 0.0
        state = tracks.get(tid)
        times = []
        if state:
            for sc, fc in zip(state.start_crossed, state.finish_crossed):
                if fc > sc: times.append(fc - sc)
        t_str = "  ".join(f"{t:.2f}s" for t in times) if times else "--"
        print(f"  ID {tid:3d}  Peak:{max(spds)*3.6:6.2f}km/h  "
              f"Avg:{np.mean(spds)*3.6:6.2f}km/h  "
              f"Dist:{dist:.1f}m  Finishes:{len(finish_events[tid])}  Splits:[{t_str}]")

    generate_speed_graphs(speed_logs, start_events, finish_events,
                          GRAPH_OUTPUT_TIME, GRAPH_OUTPUT_DIST)
    if VIDEO_OUTPUT and os.path.exists(VIDEO_OUTPUT):
        print(f"  Video -> {VIDEO_OUTPUT}")

if __name__ == "__main__":
    main()