import os, cv2, time, threading, queue, subprocess, socket
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from flask import Flask, Response

DISPLAY_SIZE   = (1280, 720)
SKIP_FRAMES    = 2        # run model every N frames (2 = skip 1, good balance)
CONF_THRESHOLD = 0.30
IOU_THRESHOLD  = 0.30
MIN_BOX_AREA   = 160
TRACK_IMGSZ    = 640      # ↓ from 960 — biggest single speed win for live streams

MODEL_PATH        = "yolov11/best.pt"
H_PATH            = "H.npy"
CALIB_PATH        = "calib.npz"
GRAPH_OUTPUT_TIME = "speed_graph_time.png"
GRAPH_OUTPUT_DIST = "speed_graph_dist.png"
VIDEO_OUTPUT      = "output_tracked.mp4"
CLASSES_TO_TRACK  = [0]
MEDIAN_WIN        = 31
LOG_EVERY_N       = 1

RTMP_INGEST_URL = "rtmp://192.168.1.17:1935/live"   
FLASK_PORT      = 5000
FLASK_ENABLED   = True

MEDIAMTX_EXE = "mediamtx.exe"
MEDIAMTX_YML = "mediamtx.yml"
RTMP_PORT    = 1935
RTSP_PORT    = 8554

CORNER_LABELS = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
EDGE_NAMES    = ["Top (TL->TR)", "Right (TR->BR)", "Bottom (BR->BL)", "Left (BL->TL)"]
EDGES         = [(0,1),(1,2),(2,3),(3,0)]
COLORS        = [(0,255,255),(0,200,255),(0,150,255),(0,100,255)]

MIN_GRAPH_FRAMES    = 10
MIN_GRAPH_SPEED_KMH = 1.0
MIN_GRAPH_DIST_M    = 0.5


class ThreadedCapture:
    def __init__(self, source):
        self._cap    = self._open(source)
        self._frame  = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _open(self, source):
        is_rtmp = isinstance(source, str) and source.startswith("rtmp://")
        if is_rtmp:
            # Use FFMPEG backend with low-latency flags via URL options
            url = (source
                   + "?buffer_size=65536"
                   + "&rtmp_live=live"
                   + "&fflags=nobuffer"
                   + "&flags=low_delay"
                   + "&max_delay=0")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"  [Capture] RTMP opened: {source}")
                return cap
            raise RuntimeError(f"Cannot open RTMP stream: {source}")

        backends = [cv2.CAP_MSMF, cv2.CAP_FFMPEG, cv2.CAP_DSHOW, cv2.CAP_ANY]
        for attempt in range(1, 4):
            for backend in backends:
                cap = cv2.VideoCapture(source, backend)
                if cap.isOpened():
                    print(f"  [Capture] Opened (attempt {attempt}, backend {backend})")
                    return cap
                cap.release()
            time.sleep(0.4)
        raise RuntimeError(f"Cannot open: {source}")

    def _reader(self):
        """Continuously decode frames; keep only the latest."""
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame   # overwrite — old frames are discarded

    def read(self):
        """Return the freshest decoded frame, or (False, None) if none yet."""
        with self._lock:
            f = self._frame
            self._frame = None        # consume it
        if f is None:
            return False, None
        return True, f

    def get_fps(self):
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._cap.release()


class MediaMTXManager:
    def __init__(self):
        self._proc = None

    def _kill_port(self, port):
        try:
            out = subprocess.check_output(
                f'netstat -ano | findstr :{port}', shell=True, text=True)
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
            return False
        print("  [MediaMTX] Clearing ports...")
        self._kill_port(RTMP_PORT)
        self._kill_port(RTSP_PORT)
        print("  [MediaMTX] Starting...")
        self._proc = subprocess.Popen(
            [MEDIAMTX_EXE, MEDIAMTX_YML],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if self._port_ready(RTMP_PORT):
            print(f"  [MediaMTX] Ready — RTMP :{RTMP_PORT}")
            return True
        print("  [MediaMTX] Timed out.")
        self.stop()
        return False

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try: self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired: self._proc.kill()
            print("  [MediaMTX] Stopped.")
        self._proc = None


def load_calibration(calib_path):
    if calib_path is None or not os.path.exists(calib_path):
        print(f"  [Calib] Not found / disabled — skipping undistortion.")
        return None, None
    data = np.load(calib_path)
    if "map1" in data and "map2" in data:
        print(f"  [Calib] Loaded maps from {calib_path}")
        return data["map1"], data["map2"]
    if "camera_matrix" in data and "dist_coeffs" in data:
        K, dist = data["camera_matrix"], data["dist_coeffs"]
        w, h = data["image_size"].astype(int)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_16SC2)
        print(f"  [Calib] Computed maps from {calib_path}")
        return map1, map2
    print("  [Calib] Unrecognised format — skipping.")
    return None, None


def undistort_frame(frame, map1, map2):
    if map1 is None: return frame
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


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


class FlaskStreamer:
    def __init__(self, port):
        self._port  = port
        self._frame = None
        self._lock  = threading.Lock()
        self._app   = Flask(__name__)
        self._app.add_url_rule("/",       "index",  self._index)
        self._app.add_url_rule("/stream", "stream", self._stream)
        threading.Thread(target=self._serve, daemon=True).start()
        print(f"  [Flask] http://0.0.0.0:{self._port}")

    def push(self, frame):
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with self._lock:
            self._frame = buf.tobytes()

    def _generate(self):
        while True:
            with self._lock: data = self._frame
            if data:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            time.sleep(0.033)

    def _stream(self):
        return Response(self._generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def _index(self):
        return ("<html><head><title>Speed Tracker</title>"
                "<style>body{background:#111;margin:0;display:flex;justify-content:center;"
                "align-items:center;height:100vh;}img{max-width:100%;border:2px solid #0f0;}"
                "</style></head><body><img src='/stream'></body></html>")

    def _serve(self):
        import logging; logging.getLogger("werkzeug").setLevel(logging.ERROR)
        self._app.run(host="0.0.0.0", port=self._port, threaded=True)


class AsyncWriter:
    def __init__(self, path, fps, size):
        self._writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self._q = queue.Queue(maxsize=128)
        threading.Thread(target=self._run, daemon=True).start()

    def write(self, frame):
        try: self._q.put_nowait(frame.copy())
        except queue.Full: pass   # drop frame rather than block

    def _run(self):
        while True:
            f = self._q.get()
            if f is None: break
            self._writer.write(f)

    def release(self):
        self._q.put(None)
        self._writer.release()


def gui_select_mode():
    result = {"mode": None}
    root = tk.Tk()
    root.title("Athlete Speed Tracker")
    root.geometry("340x220")
    root.resizable(False, False)

    def set_mode(m):
        result["mode"] = m
        root.destroy()

    tk.Label(root, text="Athlete Speed Tracker", font=("Arial", 15, "bold")).pack(pady=12)
    tk.Label(root, text="Select input source:").pack()
    tk.Button(root, text="Offline Video File",     width=26,
              command=lambda: set_mode("offline")).pack(pady=6)
    tk.Button(root, text="Real-Time Camera (RTMP)", width=26,
              command=lambda: set_mode("realtime")).pack(pady=6)
    root.mainloop()

    if result["mode"] is None: return None, None

    if result["mode"] == "offline":
        root2 = tk.Tk(); root2.withdraw()
        path = filedialog.askopenfilename(
            parent=root2, title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4"),
                       ("All files", "*.*")])
        root2.destroy()
        if not path: return None, None
        return "offline", path

    return "realtime", RTMP_INGEST_URL


def gui_pick_corners(frame):
    pts = []; clone = frame.copy()
    WIN = "FIELD CORNERS: TL>TR>BR>BL  |  BACKSPACE=undo  ENTER=confirm  ESC=cancel"

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y)); idx = len(pts) - 1
            cv2.circle(clone, (x, y), 8, COLORS[idx], -1)
            cv2.putText(clone, CORNER_LABELS[idx], (x+10, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[idx], 2)
            if len(pts) > 1: cv2.line(clone, pts[-2], pts[-1], (0,255,180), 2)
            if len(pts) == 4: cv2.line(clone, pts[-1], pts[0], (0,255,180), 2)

    cv2.destroyAllWindows(); cv2.waitKey(1)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, *DISPLAY_SIZE)
    cv2.imshow(WIN, clone)
    for _ in range(10): cv2.waitKey(30)
    cv2.setMouseCallback(WIN, on_click)

    while True:
        disp = clone.copy()
        lbl = CORNER_LABELS[len(pts)] if len(pts) < 4 else "All 4 — press ENTER"
        cv2.putText(disp, f"Next: {lbl}  |  {len(pts)}/4  |  ENTER=confirm  ESC=cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow(WIN, disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, ord('\r')) and len(pts) == 4: break
        if key == 27: pts.clear(); break
        if key == 8 and pts:
            pts.pop(); clone = frame.copy()
            for i, p in enumerate(pts):
                cv2.circle(clone, p, 8, COLORS[i], -1)
                cv2.putText(clone, CORNER_LABELS[i], (p[0]+10, p[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[i], 2)
            for i in range(1, len(pts)):
                cv2.line(clone, pts[i-1], pts[i], (0,255,180), 2)

    cv2.destroyWindow(WIN); cv2.waitKey(1)
    return pts if len(pts) == 4 else None


def gui_pick_start_finish():
    result = {"val": None}
    root = tk.Tk()
    root.title("Start / Finish Edges & Distance")
    root.geometry("400x260"); root.resizable(False, False)
    tk.Label(root, text="Choose Start & Finish Lines",
             font=("Arial", 13, "bold")).pack(pady=10)
    tk.Label(root, text="Corners: TL -> TR -> BR -> BL").pack()
    frm = tk.Frame(root); frm.pack(pady=8)
    tk.Label(frm, text="START edge:").grid(row=0, column=0, padx=8, sticky="e")
    sv = tk.StringVar(value=EDGE_NAMES[3])
    ttk.Combobox(frm, textvariable=sv, values=EDGE_NAMES,
                 state="readonly", width=22).grid(row=0, column=1)
    tk.Label(frm, text="FINISH edge:").grid(row=1, column=0, padx=8, pady=6, sticky="e")
    fv = tk.StringVar(value=EDGE_NAMES[1])
    ttk.Combobox(frm, textvariable=fv, values=EDGE_NAMES,
                 state="readonly", width=22).grid(row=1, column=1)
    tk.Label(frm, text="Distance (meters):").grid(row=2, column=0, padx=8, pady=6, sticky="e")
    dv = tk.StringVar(value="10.0")
    tk.Entry(frm, textvariable=dv, width=24).grid(row=2, column=1)

    def confirm():
        si, fi = EDGE_NAMES.index(sv.get()), EDGE_NAMES.index(fv.get())
        if si == fi:
            messagebox.showwarning("Invalid", "Start and Finish must be different."); return
        try:
            d = float(dv.get())
            if d <= 0: raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid", "Distance must be positive."); return
        result["val"] = (si, fi, d); root.destroy()

    tk.Button(root, text="Confirm", width=14, command=confirm).pack(side="left",  padx=40, pady=8)
    tk.Button(root, text="Cancel",  width=14, command=root.destroy).pack(side="right", padx=40, pady=8)
    root.mainloop()
    return result["val"]


def show_field_preview(frame, corners):
    preview = frame.copy()
    cv2.polylines(preview, [np.array(corners, np.int32)], True, (0,255,180), 2)
    for i, (ei, ej) in enumerate(EDGES):
        mx = (corners[ei][0]+corners[ej][0])//2
        my = (corners[ei][1]+corners[ej][1])//2
        cv2.putText(preview, EDGE_NAMES[i], (mx-40, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
    for i, c in enumerate(corners):
        cv2.circle(preview, c, 6, COLORS[i], -1)
        cv2.putText(preview, CORNER_LABELS[i], (c[0]+8, c[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[i], 1)
    WIN = "Field preview — press any key to continue"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, *DISPLAY_SIZE)
    cv2.imshow(WIN, preview); cv2.waitKey(0); cv2.destroyWindow(WIN)


def run_field_setup(frame, H):
    corners = gui_pick_corners(frame)
    if corners is None: return None
    show_field_preview(frame, corners)
    sf = gui_pick_start_finish()
    if sf is None: return None
    si, fi, distance_m = sf
    i0,i1 = EDGES[si]; j0,j1 = EDGES[fi]
    start_seg  = (corners[i0], corners[i1])
    finish_seg = (corners[j0], corners[j1])
    mask = np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0]), np.uint8)
    cv2.fillPoly(mask, [np.array(corners, np.int32)], 255)
    return corners, start_seg, finish_seg, mask, distance_m


def segments_cross(p1, p2, q1, q2):
    def cross2d(a, b): return a[0]*b[1] - a[1]*b[0]
    d1 = (p2[0]-p1[0], p2[1]-p1[1]); d2 = (q2[0]-q1[0], q2[1]-q1[1])
    denom = cross2d(d1, d2)
    if abs(denom) < 1e-9: return False
    t = cross2d((q1[0]-p1[0], q1[1]-p1[1]), d2) / denom
    u = cross2d((q1[0]-p1[0], q1[1]-p1[1]), d1) / denom
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def point_in_polygon(px, py, corners):
    return cv2.pointPolygonTest(
        np.array(corners, np.float32), (float(px), float(py)), False) >= 0


def make_kalman(fps, wx, wy):
    dt = 1.0 / fps
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
    kf.H = np.eye(2, 4, dtype=float)
    kf.x = np.array([[wx],[wy],[0.],[0.]])
    kf.P = np.diag([2., 2., 1., 1.])
    kf.R = np.diag([0.25, 0.25])
    kf.Q = np.diag([0.01, 0.01, 0.1, 0.1])
    return kf


class TrackState:
    __slots__ = ("kf","prev_t","prev_px","prev_wm","speed_buf","last_speed_ms",
                 "color","start_crossed","finish_crossed","total_dist",
                 "last_seen_t","last_box","last_disp_kh","roi_entry_t","roi_exit_t")

    def __init__(self, kf, t, px, py, wx_m, wy_m, color, box):
        self.kf            = kf
        self.prev_t        = t
        self.prev_px       = (px, py)
        self.prev_wm       = (wx_m, wy_m)
        self.speed_buf     = deque(maxlen=MEDIAN_WIN)
        self.last_speed_ms = 0.0
        self.color         = color
        self.start_crossed = []; self.finish_crossed = []
        self.total_dist    = 0.0
        self.last_seen_t   = t
        self.last_box      = box
        self.last_disp_kh  = 0.0
        self.roi_entry_t   = t
        self.roi_exit_t    = None


def generate_speed_graphs(speed_logs, start_events, finish_events,
                          out_time, out_dist, vid_fps=30.0, runner_names=None):
    if not speed_logs: print("No speed data."); return
    from scipy.ndimage import gaussian_filter1d

    valid_logs = {
        tid: recs for tid, recs in speed_logs.items()
        if (len(recs) >= MIN_GRAPH_FRAMES
            and np.mean([r[1]*3.6 for r in recs]) >= MIN_GRAPH_SPEED_KMH
            and (recs[-1][2] if len(recs[-1]) > 2 else 0.) >= MIN_GRAPH_DIST_M)
    }
    if not valid_logs: print("No valid speed data after filtering."); return
    print(f"  [Graph] Plotting {len(valid_logs)} tracks")
    palette = plt.cm.tab20.colors

    for out_path, x_key, xlabel, title_sfx in [
        (out_time, "time", "Time (s)",     "Over Time"),
        (out_dist, "dist", "Distance (m)", "Over Distance"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title(f"Athlete Speed Profiles — {title_sfx}",
                     fontsize=14, fontweight="bold", pad=12)
        all_xs = []
        for tid, records in sorted(valid_logs.items()):
            if len(records) < 3: continue
            ts   = np.array([r[0] for r in records])
            kmph = np.array([r[1]*3.6 for r in records])
            ds   = np.array([r[2] if len(r)>2 else 0. for r in records])
            xs   = (ts - ts[0]) if x_key == "time" else ds
            col  = palette[tid % len(palette)]
            order = np.argsort(xs)
            xs_s, kmph_s = xs[order], kmph[order]
            all_xs.extend(xs_s.tolist())
            sigma  = max(3, int(vid_fps / 6))
            smooth = gaussian_filter1d(kmph_s, sigma=sigma)
            peak, avg = float(np.max(kmph_s)), float(np.mean(kmph_s))
            name_label = f"ID {tid}"
            if runner_names and tid in runner_names and runner_names[tid]:
                name_label = f"{runner_names[tid]} (ID {tid})"
            ax.plot(xs_s, smooth, color=col, linewidth=2.2,
                    label=f"{name_label}  |  peak {peak:.1f} km/h  avg {avg:.1f} km/h")
            ax.plot(xs_s[-1], smooth[-1], 'x', color=col, ms=8, mew=2.5, zorder=5)
            if len(smooth):
                pk = int(np.argmax(smooth))
                x_span   = float(xs_s[-1]-xs_s[0]) if len(xs_s)>1 else 1.
                offset_x = max(x_span*0.04, 0.5)
                if xs_s[pk] > xs_s[0]+x_span*0.55: offset_x = -abs(offset_x)-1.
                ax.annotate(f"{name_label}\n{smooth[pk]:.1f} km/h",
                            xy=(xs_s[pk], smooth[pk]),
                            xytext=(xs_s[pk]+offset_x, smooth[pk]+1.5),
                            fontsize=7, color=col, fontweight="bold",
                            ha="right" if offset_x<0 else "left",
                            arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
            t0 = float(ts[0])
            for st in start_events.get(tid, []):
                idx = int(np.searchsorted(ts, st))
                xv  = (st-t0) if x_key=="time" else ds[min(idx, len(ds)-1)]
                ax.axvline(xv, color=col, ls=":", lw=1.2, alpha=0.7)
            for ft in finish_events.get(tid, []):
                idx = int(np.searchsorted(ts, ft))
                xv  = (ft-t0) if x_key=="time" else ds[min(idx, len(ds)-1)]
                ax.axvline(xv, color=col, ls="--", lw=1.2, alpha=0.7)

        if all_xs:
            margin = max((max(all_xs)-min(all_xs))*0.03, 0.5)
            ax.set_xlim(min(all_xs)-margin, max(all_xs)+margin)
        ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel("Speed (km/h)", fontsize=11)
        ax.set_ylim(bottom=0); ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.88,
                  title="Subject  (dotted=start  dashed=finish  x=last seen)",
                  title_fontsize=8)
        plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(); print(f"  Graph saved -> {out_path}")


class StartGate:
    """Tracks whether the user has clicked the on-screen start button."""
    def __init__(self):
        self.active    = False   # False = waiting, True = recording
        self.clicked_t = None    # wall time of click

    # Call every frame to draw the button and handle a mouse click
    def draw_and_check(self, frame, mouse_xy):
        """
        Draws button onto frame (in-place).
        mouse_xy: latest (x,y) from the mouse callback, or None.
        Returns True on the frame the button is first clicked.
        """
        if self.active:
            # Already started — draw a small green indicator
            cv2.rectangle(frame, (10, 50), (220, 80), (0,60,0), -1)
            cv2.putText(frame, "● DETECTION ACTIVE", (16, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,100), 2, cv2.LINE_AA)
            return False

        # Big green button
        bx1, by1, bx2, by2 = 10, 50, 310, 95
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 160, 0), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0),  2)
        cv2.putText(frame, "▶  START DETECTION", (bx1+12, by2-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        if mouse_xy is not None:
            mx, my = mouse_xy
            if bx1 <= mx <= bx2 and by1 <= my <= by2:
                self.active    = True
                self.clicked_t = time.perf_counter()
                return True
        return False


def main():
    mode, source = gui_select_mode()
    if mode is None: print("Cancelled."); return
    print(f"\nMode: {mode}  |  Source: {source}")

    mediamtx = None
    if mode == "realtime":
        mediamtx = MediaMTXManager()
        if not mediamtx.start():
            print("  [MediaMTX] Failed — continuing anyway.")

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():           device = "cuda"
        elif torch.backends.mps.is_available(): device = "mps"
    except Exception: pass
    use_half = device in ("cuda", "mps")
    print(f"Device: {device.upper()}  fp16: {use_half}")

    calib_map1, calib_map2 = load_calibration(CALIB_PATH)
    flask_streamer = FlaskStreamer(FLASK_PORT) if FLASK_ENABLED else None

    print(f"Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    H     = load_homography(H_PATH)
    H_inv = np.linalg.inv(H)

    if mode == "realtime":
        print("\n  Waiting for DJI stream on RTMP...")
        print(f"  -> DJI Fly: Transmission → Live Streaming → Custom RTMP")
        print(f"  -> URL: {RTMP_INGEST_URL}")
        print("  -> Tap Start in DJI Fly, THEN press Enter here.")
        input("  Press Enter when drone is streaming...\n")

    # Grab first frame for setup
    print("Grabbing first frame for setup...")
    # Use a simple blocking cap just for the setup frame
    setup_cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    setup_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    first_frame = None
    for _ in range(30):
        ret, f = setup_cap.read()
        if ret and f is not None:
            first_frame = cv2.resize(undistort_frame(f, calib_map1, calib_map2), DISPLAY_SIZE)
            break
        time.sleep(0.1)
    setup_cap.release()
    if first_frame is None:
        print("ERROR: Could not grab a frame. Check stream."); return

    setup = run_field_setup(first_frame, H)
    if setup is None:
        print("Field setup cancelled.")
        if mediamtx: mediamtx.stop()
        return
    corners, start_seg, finish_seg, roi_mask, distance_m = setup

    # Scale
    scx = (start_seg[0][0]+start_seg[1][0])/2.
    scy = (start_seg[0][1]+start_seg[1][1])/2.
    fcx = (finish_seg[0][0]+finish_seg[1][0])/2.
    fcy = (finish_seg[0][1]+finish_seg[1][1])/2.
    ws_x,ws_y = pixel_to_world(H, scx, scy)
    wf_x,wf_y = pixel_to_world(H, fcx, fcy)
    world_dist = np.hypot(wf_x-ws_x, wf_y-ws_y)
    scale_m_per_w = distance_m/world_dist if world_dist > 0 else 1.0
    print(f"  [Geometry] {distance_m}m = {world_dist:.2f} wu  (scale {scale_m_per_w:.4f} m/wu)")

    print("Opening threaded capture...")
    cap     = ThreadedCapture(source)
    vid_fps = cap.get_fps()
    print(f"FPS: {vid_fps:.2f}")

    writer     = AsyncWriter(VIDEO_OUTPUT, vid_fps, DISPLAY_SIZE) if VIDEO_OUTPUT else None
    start_gate = StartGate()

    tracks          = {}
    finished_tracks = {}
    id_remap        = {}
    speed_logs      = defaultdict(list)
    start_events    = defaultdict(list)
    finish_events   = defaultdict(list)

    def rand_color(tid):
        hue = (tid * 0.618033988749895) % 1.0
        hsv = np.uint8([[[int(hue*179), 220, 210]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    # Mouse callback for on-screen start button 
    WIN_NAME   = "Speed Tracker"
    mouse_pos  = [None]   # mutable container so closure can write to it
    paused     = False

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_pos[0] = (x, y)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, *DISPLAY_SIZE)
    cv2.setMouseCallback(WIN_NAME, on_mouse)

    frame_idx       = 0
    fps_display     = 0.0
    t_fps           = time.perf_counter()
    last_det_result = None

    # For RTMP: prime the threaded reader for ~1s to flush the initial buffer
    if mode == "realtime":
        print("  Flushing RTMP buffer (1s)...")
        time.sleep(1.0)

    print("\nTracking — ESC=stop" +
          ("  R=recalibrate  Click green button to start recording" if mode=="realtime" else "") + "\n")

    while True:
        # Recalibrate
        if paused:
            # grab a fresh live frame
            live = None
            for _ in range(30):
                ret, f = cap.read()
                if ret and f is not None:
                    live = cv2.resize(undistort_frame(f, calib_map1, calib_map2), DISPLAY_SIZE)
                    break
                time.sleep(0.05)
            if live is not None:
                new_setup = run_field_setup(live, H)
                if new_setup is not None:
                    corners, start_seg, finish_seg, roi_mask, distance_m = new_setup
                    scx = (start_seg[0][0]+start_seg[1][0])/2.
                    scy = (start_seg[0][1]+start_seg[1][1])/2.
                    fcx = (finish_seg[0][0]+finish_seg[1][0])/2.
                    fcy = (finish_seg[0][1]+finish_seg[1][1])/2.
                    ws_x,ws_y = pixel_to_world(H, scx, scy)
                    wf_x,wf_y = pixel_to_world(H, fcx, fcy)
                    world_dist = np.hypot(wf_x-ws_x, wf_y-ws_y)
                    scale_m_per_w = distance_m/world_dist if world_dist > 0 else 1.0
                    tracks.clear()
                    start_gate.active = False   # reset gate on recalibrate
            # Reattach mouse callback after OpenCV windows were recreated
            cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_NAME, *DISPLAY_SIZE)
            cv2.setMouseCallback(WIN_NAME, on_mouse)
            paused = False

        # Grab latest frame
        ret, frame = cap.read()
        if not ret:
            # No new frame yet — show last annotated or wait
            cv2.waitKey(5)
            continue

        frame = undistort_frame(frame, calib_map1, calib_map2)
        frame = cv2.resize(frame, DISPLAY_SIZE)
        frame_idx += 1
        t_now = (frame_idx / vid_fps) if mode == "offline" else time.perf_counter()

        # Model inference (skip frames to maintain display FPS)
        if (frame_idx % SKIP_FRAMES == 0) or frame_idx == 1:
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=CLASSES_TO_TRACK,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                half=use_half,
                device=device,
                verbose=False,
                imgsz=TRACK_IMGSZ,
            )
            last_det_result = results[0]

        if last_det_result is None:
            cv2.imshow(WIN_NAME, frame)
            if cv2.waitKey(1) == 27: break
            continue

        # Expire old finished tracks
        FINISHED_TTL = 6.0
        for _tid in [k for k,s in finished_tracks.items()
                     if t_now - s.last_seen_t > FINISHED_TTL]:
            del finished_tracks[_tid]

        annotated   = frame.copy()
        current_ids = set()

        det = last_det_result
        if det.boxes is not None and det.boxes.id is not None:
            boxes = det.boxes.xyxy.cpu().numpy()
            ids   = det.boxes.id.cpu().numpy().astype(int)
            confs = det.boxes.conf.cpu().numpy()
            current_frame_ctids = {id_remap.get(t,t) for t in ids}

            for box, tid, conf in zip(boxes, ids, confs):
                x1,y1,x2,y2 = box
                if (x2-x1)*(y2-y1) < MIN_BOX_AREA: continue
                cx_px = (x1+x2)*0.5
                cy_px = float(y2)
                if not point_in_polygon(int(cx_px), int(cy_px), corners): continue

                ctid = id_remap.get(tid, tid)
                current_ids.add(ctid)
                if ctid in tracks and tracks[ctid].roi_exit_t is not None:
                    tracks[ctid].roi_exit_t = None

                wx,wy  = pixel_to_world(H, cx_px, cy_px)
                wx_m   = wx * scale_m_per_w
                wy_m   = wy * scale_m_per_w

                if tid not in tracks and ctid not in tracks:
                    REID_DIST_M = 5.0
                    best_tid, best_dist, best_fin = None, REID_DIST_M, False
                    candidates = ([(k,v,True)  for k,v in finished_tracks.items()] +
                                  [(k,v,False) for k,v in tracks.items()
                                   if k not in current_frame_ctids])
                    for old_tid, old_s, is_fin in candidates:
                        dt_s   = t_now - old_s.last_seen_t
                        pred_x = float(old_s.kf.x[0]) + float(old_s.kf.x[2])*dt_s
                        pred_y = float(old_s.kf.x[1]) + float(old_s.kf.x[3])*dt_s
                        d = np.hypot(wx_m-pred_x, wy_m-pred_y)
                        if d < best_dist: best_dist,best_tid,best_fin = d,old_tid,is_fin

                    if best_tid is not None:
                        canonical = best_tid; id_remap[tid] = canonical
                        old_s = (finished_tracks.pop(best_tid) if best_fin
                                 else tracks[best_tid])
                        old_s.kf.x[0] = wx_m; old_s.kf.x[1] = wy_m
                        old_s.prev_t = t_now; old_s.prev_wm = (wx_m, wy_m)
                        old_s.last_seen_t = t_now; old_s.roi_exit_t = None
                        old_s.last_box = (x1,y1,x2,y2)
                        tracks[canonical] = old_s
                        for src in (speed_logs, start_events, finish_events):
                            if tid in src and tid != canonical:
                                src[canonical].extend(src.pop(tid))
                                src[canonical].sort(key=lambda r: r[0])
                    else:
                        canonical = ctid
                        tracks[canonical] = TrackState(
                            kf=make_kalman(vid_fps, wx_m, wy_m),
                            t=t_now, px=int(cx_px), py=int(cy_px),
                            wx_m=wx_m, wy_m=wy_m,
                            color=rand_color(canonical),
                            box=(x1,y1,x2,y2))
                else:
                    canonical = ctid

                state = tracks[canonical]; kf = state.kf
                q_scale = 1.0 + (1.0 - float(conf)) * 2.0
                kf.Q = np.diag([0.01, 0.01, 0.08*q_scale, 0.08*q_scale])
                kf.predict(); kf.update([wx_m, wy_m])

                dt_frame   = t_now - state.prev_t
                smoothed_x = float(kf.x[0]); smoothed_y = float(kf.x[1])

                if dt_frame > 1e-6:
                    prev_x, prev_y = state.prev_wm
                    speed_ms = np.hypot(smoothed_x-prev_x, smoothed_y-prev_y) / dt_frame
                    speed_ms = min(speed_ms, 8.5)
                else:
                    speed_ms = state.last_speed_ms

                state.speed_buf.append(speed_ms)
                disp_ms = float(np.median(state.speed_buf))
                disp_kh = disp_ms * 3.6

                # Only accumulate distance/log when gate is open
                if start_gate.active and state.prev_t > 0:
                    state.total_dist += disp_ms * dt_frame

                state.last_speed_ms = speed_ms
                state.prev_t        = t_now
                state.prev_wm       = (smoothed_x, smoothed_y)
                state.last_seen_t   = t_now
                state.last_box      = (x1,y1,x2,y2)
                state.last_disp_kh  = disp_kh

                new_px,new_py = int(cx_px), int(cy_px)
                ppx,ppy = state.prev_px
                if start_gate.active:
                    if segments_cross((ppx,ppy),(new_px,new_py), start_seg[0], start_seg[1]):
                        state.start_crossed.append(t_now); start_events[canonical].append(t_now)
                    if segments_cross((ppx,ppy),(new_px,new_py), finish_seg[0], finish_seg[1]):
                        state.finish_crossed.append(t_now); finish_events[canonical].append(t_now)
                state.prev_px = (new_px, new_py)

                if start_gate.active and frame_idx % LOG_EVERY_N == 0:
                    speed_logs[canonical].append((t_now, disp_ms, state.total_dist))

        # Draw / expire tracks
        for tid in list(tracks.keys()):
            state = tracks[tid]
            if t_now - state.last_seen_t > 3.0:
                if state.roi_exit_t is None: state.roi_exit_t = state.last_seen_t
                finished_tracks[tid] = state; del tracks[tid]; continue

            col = state.color; disp_kh = state.last_disp_kh

            if tid not in current_ids:
                if state.roi_exit_t is None: state.roi_exit_t = t_now
                dt = t_now - state.prev_t
                if dt > 0:
                    state.kf.predict()
                    disp_ms = float(np.median(state.speed_buf)) if state.speed_buf else 0.
                    if start_gate.active: state.total_dist += disp_ms * dt
                    state.prev_t = t_now
                pred_x = float(state.kf.x[0]); pred_y = float(state.kf.x[1])
                new_cx,new_cy = world_to_pixel(H_inv, pred_x/scale_m_per_w, pred_y/scale_m_per_w)
                ox1,oy1,ox2,oy2 = state.last_box
                bw,bh = ox2-ox1, oy2-oy1
                if bw < 4 or bh < 4: continue
                dx,dy = new_cx-(ox1+bw*0.5), new_cy-float(oy2)
                bx1,by1,bx2,by2 = ox1+dx,oy1+dy,ox2+dx,oy2+dy
                state.last_box = (bx1,by1,bx2,by2)
                draw_col = tuple(max(0,c-60) for c in col)
                if t_now - state.last_seen_t < 1.0:
                    cv2.rectangle(annotated,(int(bx1),int(by1)),(int(bx2),int(by2)),
                                  draw_col, 2, cv2.LINE_4)
            else:
                bx1,by1,bx2,by2 = state.last_box
                bw,bh = bx2-bx1, by2-by1
                if bw < 4 or bh < 4: continue
                draw_col = col
                cv2.rectangle(annotated,(int(bx1),int(by1)),(int(bx2),int(by2)),draw_col,2)

            if tid in current_ids or (t_now - state.last_seen_t < 1.0):
                label = f"ID {tid}  {disp_kh:.1f}km/h  {state.total_dist:.1f}m"
                (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                lx = min(max(int(bx1),0), annotated.shape[1]-tw-8)
                ly = max(int(by1)-8, th+4)
                cv2.rectangle(annotated,(lx,ly-th-4),(lx+tw+6,ly+2),draw_col,-1)
                cv2.putText(annotated, label, (lx+3,ly-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2, cv2.LINE_AA)

                if state.start_crossed and not state.finish_crossed:
                    elapsed = t_now - state.start_crossed[-1]
                    cv2.putText(annotated, f"SPRINT {elapsed:.1f}s",
                                (int(bx1), max(int(by1)-30,20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)
                elif state.finish_crossed:
                    cv2.putText(annotated, "FINISH",
                                (int(bx1), max(int(by1)-30,20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,80,255), 2, cv2.LINE_AA)

                if state.roi_exit_t is None:
                    roi_e = t_now - state.roi_entry_t
                    t_txt,t_col = f"In RoI: {roi_e:.2f}s", (0,255,180)
                else:
                    roi_e = state.roi_exit_t - state.roi_entry_t
                    t_txt,t_col = f"RoI: {roi_e:.2f}s (done)", (180,180,180)
                cv2.putText(annotated, t_txt,
                            (int(bx1), min(int(by2)+18, annotated.shape[0]-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, t_col, 1, cv2.LINE_AA)

        # HUD
        t_wall      = time.perf_counter()
        fps_display = 0.9*fps_display + 0.1/max(t_wall-t_fps, 1e-9)
        t_fps       = t_wall
        hint = "ESC=stop  R=recalibrate" if mode=="realtime" else "ESC=stop"
        cv2.putText(annotated,
                    f"FPS:{fps_display:.1f}  Frame:{frame_idx}  [{device.upper()}]  {hint}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2, cv2.LINE_AA)

        # On-screen start button (realtime only)
        if mode == "realtime":
            clicked = start_gate.draw_and_check(annotated, mouse_pos[0])
            if clicked:
                print("  [Gate] Detection started — recording speeds.")
            mouse_pos[0] = None   # consume the click

        # Global RoI clock
        all_states = list(tracks.values()) + list(finished_tracks.values())
        entered    = [s for s in all_states if s.roi_entry_t is not None]
        active_s   = [s for s in tracks.values() if s.roi_exit_t is None]
        if entered and start_gate.active:
            g_start = min(s.roi_entry_t for s in entered)
            if active_s:
                g_elapsed  = t_now - g_start
                clock_col  = (0,255,100)
                clock_lbl  = f"RoI Clock: {g_elapsed:.2f}s"
            else:
                g_elapsed  = max(s.roi_exit_t for s in entered) - g_start
                clock_col  = (100,200,255)
                clock_lbl  = f"RoI Clock: {g_elapsed:.2f}s  [all out]"
            (cw,ch),_ = cv2.getTextSize(clock_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cx0 = annotated.shape[1]//2 - cw//2
            cy0 = annotated.shape[0] - 44
            ov  = annotated.copy()
            cv2.rectangle(ov,(cx0-10,cy0-ch-6),(cx0+cw+10,cy0+6),(20,20,20),-1)
            cv2.addWeighted(ov,0.55,annotated,0.45,0,annotated)
            cv2.putText(annotated,clock_lbl,(cx0,cy0),
                        cv2.FONT_HERSHEY_SIMPLEX,0.75,clock_col,2,cv2.LINE_AA)

        # Draw RoI polygon
        cv2.polylines(annotated,[np.array(corners,np.int32)],True,(0,200,100),1)

        if flask_streamer: flask_streamer.push(annotated)
        cv2.imshow(WIN_NAME, annotated)
        if writer: writer.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        if key in (ord('r'),ord('R')) and mode == "realtime":
            print("  R — recalibrating..."); paused = True

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    if mediamtx: mediamtx.stop()

    # Summary
    print("\n--- Summary ---")
    all_logged = {**tracks, **finished_tracks}
    for tid in sorted(speed_logs):
        recs = speed_logs[tid]
        if not recs: continue
        spds = [r[1] for r in recs]
        dist = recs[-1][2] if len(recs[-1]) > 2 else 0.
        state = all_logged.get(tid)
        times = []
        if state:
            for sc,fc in zip(state.start_crossed, state.finish_crossed):
                if fc > sc: times.append(fc-sc)
        t_str = "  ".join(f"{t:.2f}s" for t in times) if times else "--"
        print(f"  ID {tid:3d}  Peak:{max(spds)*3.6:6.2f}km/h  "
              f"Avg:{np.mean(spds)*3.6:6.2f}km/h  "
              f"Dist:{dist:.1f}m  Finishes:{len(finish_events[tid])}  Splits:[{t_str}]")

    # Runner name prompt
    valid_tids = [
        tid for tid,recs in speed_logs.items()
        if (len(recs) >= MIN_GRAPH_FRAMES
            and np.mean([r[1]*3.6 for r in recs]) >= MIN_GRAPH_SPEED_KMH
            and (recs[-1][2] if len(recs[-1])>2 else 0.) >= MIN_GRAPH_DIST_M)
    ]
    runner_names = {}
    if valid_tids:
        root3 = tk.Tk(); root3.title("Enter Runner Names")
        root3.geometry("400x400"); root3.resizable(True, True)
        tk.Label(root3, text="Enter runner names (optional):",
                 wraplength=380, font=("Arial",11)).pack(pady=10)
        canvas    = tk.Canvas(root3)
        scrollbar = ttk.Scrollbar(root3, orient="vertical", command=canvas.yview)
        scr_frame = ttk.Frame(canvas)
        scr_frame.bind("<Configure>",
                        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=scr_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        entries = {}
        for tid_val in sorted(valid_tids):
            rf = tk.Frame(scr_frame); rf.pack(fill="x", padx=10, pady=5)
            tk.Label(rf, text=f"ID {tid_val}:", width=8, anchor="e").pack(side="left")
            ent = tk.Entry(rf, width=30); ent.pack(side="left", padx=5)
            entries[tid_val] = ent
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def save_names():
            for t,ent in entries.items():
                v = ent.get().strip()
                if v: runner_names[t] = v
            root3.destroy()

        bf = tk.Frame(root3); bf.pack(fill="x", pady=10)
        tk.Button(bf, text="Generate Graphs", command=save_names,
                  width=20, bg="#4CAF50", fg="white",
                  font=("Arial",10,"bold")).pack()
        root3.mainloop()

    generate_speed_graphs(speed_logs, start_events, finish_events,
                          GRAPH_OUTPUT_TIME, GRAPH_OUTPUT_DIST,
                          vid_fps, runner_names=runner_names)
    if VIDEO_OUTPUT and os.path.exists(VIDEO_OUTPUT):
        print(f"  Video -> {VIDEO_OUTPUT}")


if __name__ == "__main__":
    main()