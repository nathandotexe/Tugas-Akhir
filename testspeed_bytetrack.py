import os, cv2, time, threading, queue
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

DISPLAY_SIZE      = (1280, 720)
SKIP_FRAMES       = 1

CONF_THRESHOLD    = 0.30
IOU_THRESHOLD     = 0.30
MIN_BOX_AREA      = 160
TRACK_IMGSZ       = 960
FLASK_ENABLED     = False
VIDEO_OUTPUT      = None

MODEL_PATH        = "yolov11/best.pt"
H_PATH            = "H.npy"
CALIB_PATH        = "calib.npz"
GRAPH_OUTPUT_TIME = "speed_graph_time.png"
GRAPH_OUTPUT_DIST = "speed_graph_dist.png"
VIDEO_OUTPUT      = "output_tracked.mp4"
CLASSES_TO_TRACK  = [0]
MEDIAN_WIN        = 45  # wider window = more spike rejection
LOG_EVERY_N       = 1

RTMP_URL          = "rtmp://192.168.1.17:1935/live"
FLASK_PORT        = 5000
FLASK_ENABLED     = True

SHOW_SPEED_OVERLAY = True   # show km/h column in the top-right info panel
SHOW_ROI_OVERLAY   = True   # show RoI timer column in the top-right info panel

CORNER_LABELS = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
EDGE_NAMES    = ["Top (TL->TR)", "Right (TR->BR)", "Bottom (BR->BL)", "Left (BL->TL)"]
EDGES         = [(0,1),(1,2),(2,3),(3,0)]
COLORS        = [(0,255,255),(0,200,255),(0,150,255),(0,100,255)]

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
    is_rtmp = isinstance(source, str) and source.startswith("rtmp://")
    if is_rtmp:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"  [RTMP] Stream opened: {source}")
            return cap
        raise RuntimeError(f"Cannot open RTMP stream: {source}")
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

    source = result["path"] if result["mode"] == "offline" else RTMP_URL
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
    root.geometry("400x280")
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
    dist_var = tk.StringVar(value="")
    tk.Entry(frm, textvariable=dist_var, width=24).grid(row=2, column=1)
    tk.Label(frm, text="(leave blank for pixel-only mode)",
             font=("Arial", 8), fg="gray").grid(row=3, column=1, sticky="w")

    def confirm():
        si = EDGE_NAMES.index(start_var.get())
        fi = EDGE_NAMES.index(finish_var.get())
        if si == fi:
            messagebox.showwarning("Invalid", "Start and Finish must be different.")
            return
        raw = dist_var.get().strip()
        if raw == "":
            dist_val = None  # pure pixel mode
        else:
            try:
                dist_val = float(raw)
                if dist_val <= 0: raise ValueError
            except ValueError:
                messagebox.showwarning("Invalid", "Distance must be a positive number (or blank for pixel mode).")
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
    if not path or not os.path.exists(path):
        print(f"  [Homography] Not found: '{path}' — running in pixel-scale mode.")
        return None
    return np.load(path)


def build_pixel_homography(scale_px_to_m):
    H = np.eye(3, dtype=np.float64)
    H[0, 0] = scale_px_to_m
    H[1, 1] = scale_px_to_m
    return H


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
    kf.P = np.diag([2.0, 2.0, 1.0, 1.0])

    # ~5-10px; at ~0.05 m/px that's ~0.25-0.5 m. Use 0.5² = 0.25 per axis.
    kf.R = np.diag([0.25, 0.25])
    # Q: process noise — runners accelerate at most ~2-3 m/s² so keep this small
    kf.Q = np.diag([0.01, 0.01, 0.1, 0.1])
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
    __slots__ = ("kf","prev_t","prev_px","prev_wm","speed_buf","last_speed_ms",
                 "color","start_crossed","finish_crossed","total_dist",
                 "last_seen_t","last_box","last_disp_kh",
                 "roi_entry_t","roi_exit_t")

    def __init__(self, kf, t, px, py, wx_m, wy_m, color, box):
        self.kf             = kf
        self.prev_t         = t
        self.prev_px        = (px, py)
        self.prev_wm        = (wx_m, wy_m)   # world-metre coords last frame
        self.speed_buf      = deque(maxlen=MEDIAN_WIN)
        self.last_speed_ms  = 0.0
        self.color          = color
        self.start_crossed  = []
        self.finish_crossed = []
        self.total_dist     = 0.0
        self.last_seen_t    = t
        self.last_box       = box
        self.last_disp_kh   = 0.0
        self.roi_entry_t    = t    # time this subject first entered the RoI
        self.roi_exit_t     = None # time this subject left the RoI (None = still inside)


# Minimum thresholds to include a track in the graph
MIN_GRAPH_FRAMES    = 10    # raised from 5 — a split fragment only exists for a few frames
MIN_GRAPH_SPEED_KMH = 1.0   # raised from 0.5 — filters truly stationary ghosts
MIN_GRAPH_DIST_M    = 0.5   # raised from 0.3 — must have physically moved


def generate_speed_graphs(speed_logs, start_events, finish_events, out_path_time, out_path_dist, vid_fps=30.0, runner_names=None, pure_pixel_mode=False):
    if not speed_logs:
        print("No speed data."); return
    from scipy.ndimage import gaussian_filter1d

    # Filter out ghost/noise tracks before plotting
    # In pure pixel mode speeds are px/s (no conversion), otherwise km/h
    filt_speed_mult = 1.0 if pure_pixel_mode else 3.6
    # Thresholds: 1 km/h ≈ 0.28 m/s ≈ ~5 px/s at typical drone altitude
    min_speed_filt = 5.0 if pure_pixel_mode else MIN_GRAPH_SPEED_KMH
    min_dist_filt  = 10.0 if pure_pixel_mode else MIN_GRAPH_DIST_M

    valid_logs = {}
    for tid, records in speed_logs.items():
        if len(records) < MIN_GRAPH_FRAMES:
            continue
        speeds_conv = [r[1] * filt_speed_mult for r in records]
        dist = records[-1][2] if len(records[-1]) > 2 else 0.0
        if float(np.mean(speeds_conv)) < min_speed_filt:
            continue
        if dist < min_dist_filt:
            continue
        valid_logs[tid] = records

    if not valid_logs:
        print("No valid speed data after filtering."); return
    print(f"  [Graph] Plotting {len(valid_logs)} tracks "
          f"(filtered {len(speed_logs) - len(valid_logs)} ghost/noise tracks)")

    # tab20 gives 20 visually distinct colours; enough for large groups
    palette = plt.cm.tab20.colors

    speed_unit = "px/s" if pure_pixel_mode else "km/h"
    dist_unit  = "px"   if pure_pixel_mode else "m"
    speed_mult = 1.0    if pure_pixel_mode else 3.6   # m/s→km/h or px/s→px/s

    for out_path, x_key, xlabel, title_suffix in [
        (out_path_time, "time", "Time (s)",                       "Over Time"),
        (out_path_dist, "dist", f"Distance ({dist_unit})", "Over Distance"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title(f"Athlete Speed Profiles — {title_suffix}",
                     fontsize=14, fontweight="bold", pad=12)

        all_xs = []

        for tid, records in sorted(valid_logs.items()):
            if len(records) < 3:
                continue

            ts   = np.array([r[0] for r in records])
            kmph = np.array([r[1] for r in records]) * speed_mult
            ds   = np.array([r[2] if len(r) > 2 else 0.0 for r in records])

            # Normalise time to RoI entry so subjects are comparable from t=0
            ts_norm = ts - ts[0]
            xs   = ts_norm if x_key == "time" else ds
            col  = palette[tid % len(palette)]

            order  = np.argsort(xs)
            xs_s   = xs[order]
            kmph_s = kmph[order]
            all_xs.extend(xs_s.tolist())

            # Faint raw trace (commented out to prevent visual confusion with ghosts)
            # ax.plot(xs_s, kmph_s, color=col, alpha=0.15, linewidth=1)
            # Use dynamic smoothing based on frame rate (a 3-sigma window roughly spans ~18 frames at 60fps)
            sigma_val = max(3, int(vid_fps / 6))
            smooth = gaussian_filter1d(kmph_s, sigma=sigma_val)
            peak   = float(np.max(kmph_s))
            avg    = float(np.mean(kmph_s))
            
            name_label = f"ID {tid}"
            if runner_names and tid in runner_names and runner_names[tid]:
                name_label = f"{runner_names[tid]} (ID {tid})"

            ax.plot(xs_s, smooth, color=col, linewidth=2.2,
                    label=f"{name_label}  |  peak {peak:.1f} {speed_unit}  avg {avg:.1f} {speed_unit}")

            # End-of-track marker so line endings are explicit, not ambiguous
            ax.plot(xs_s[-1], smooth[-1], marker='x', color=col,
                    markersize=8, markeredgewidth=2.5, zorder=5)

            # Peak annotation — flip side if peak is in the right half to avoid crowding
            if len(smooth) > 0:
                pk      = int(np.argmax(smooth))
                x_span  = float(xs_s[-1] - xs_s[0]) if len(xs_s) > 1 else 1.0
                offset_x = max(x_span * 0.04, 0.5)
                offset_y = 1.5
                if xs_s[pk] > xs_s[0] + x_span * 0.55:
                    offset_x = -abs(offset_x) - 1.0
                ann_ha = "right" if offset_x < 0 else "left"
                ann_txt = "%s\n%.1f %s" % (name_label, smooth[pk], speed_unit)
                ax.annotate(ann_txt,
                            xy=(xs_s[pk], smooth[pk]),
                            xytext=(xs_s[pk] + offset_x, smooth[pk] + offset_y),
                            fontsize=7, color=col, fontweight="bold", ha=ann_ha,
                            arrowprops=dict(arrowstyle="->", color=col, lw=0.9))

            # Start / finish crossing markers per subject
            t0 = float(ts[0])
            for st in start_events.get(tid, []):
                idx = int(np.searchsorted(ts, st))
                xv  = (st - t0) if x_key == "time" else ds[min(idx, len(ds) - 1)]
                ax.axvline(xv, color=col, linestyle=":", linewidth=1.2, alpha=0.7)
            for ft in finish_events.get(tid, []):
                idx = int(np.searchsorted(ts, ft))
                xv  = (ft - t0) if x_key == "time" else ds[min(idx, len(ds) - 1)]
                ax.axvline(xv, color=col, linestyle="--", linewidth=1.2, alpha=0.7)

        # Shared x-axis covers all subjects so nothing looks missing
        if all_xs:
            margin = max((max(all_xs) - min(all_xs)) * 0.03, 0.5)
            ax.set_xlim(left=min(all_xs) - margin, right=max(all_xs) + margin)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(f"Speed ({speed_unit})", fontsize=11)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.88,
                  title="Subject  (dotted=start  dashed=finish  x=last seen)",
                  title_fontsize=8)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Graph saved -> {out_path}")


def main():
    mode, source = gui_select_mode()
    if mode is None:
        print("Cancelled."); return
    print(f"\nMode: {mode}  |  Source: {source}")

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
    # H_inv and pixel-mode flag are resolved after field setup when we know
    # the corner pixel coordinates (needed to compute pixel scale).
    H_inv      = None
    pixel_mode = (H is None)

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
        return
    corners, start_seg, finish_seg, roi_mask, distance_m = setup

    # pure_pixel_mode: no homography AND no distance → speeds in px/s
    pure_pixel_mode = (pixel_mode and distance_m is None)

    scx = (start_seg[0][0] + start_seg[1][0]) / 2.0
    scy = (start_seg[0][1] + start_seg[1][1]) / 2.0
    fcx = (finish_seg[0][0] + finish_seg[1][0]) / 2.0
    fcy = (finish_seg[0][1] + finish_seg[1][1]) / 2.0

    if pure_pixel_mode:
        # No homography, no distance — identity H so world units = pixels
        H     = build_pixel_homography(1.0)   # 1 px = 1 wu
        H_inv = np.linalg.inv(H)
        scale_m_per_w = 1.0                   # "metres" are really pixels
        print("  [Geometry] Pure pixel mode: speeds will be in px/s")
    elif pixel_mode:
        # No homography file but distance provided — derive px→m scale
        px_dist = np.hypot(fcx - scx, fcy - scy)
        if px_dist < 1:
            px_dist = 1.0
        px_to_m = distance_m / px_dist          # metres per pixel
        H       = build_pixel_homography(px_to_m)
        H_inv   = np.linalg.inv(H)
        ws_x, ws_y = pixel_to_world(H, scx, scy)
        wf_x, wf_y = pixel_to_world(H, fcx, fcy)
        world_dist = np.hypot(wf_x - ws_x, wf_y - ws_y)
        scale_m_per_w = distance_m / world_dist if world_dist > 0 else 1.0
        print(f"  [Geometry] Pixel mode: {distance_m}m over {px_dist:.1f} px "
              f"→ scale {px_to_m:.4f} m/px")
    else:
        H_inv = np.linalg.inv(H)
        ws_x, ws_y = pixel_to_world(H, scx, scy)
        wf_x, wf_y = pixel_to_world(H, fcx, fcy)
        world_dist = np.hypot(wf_x - ws_x, wf_y - ws_y)
        scale_m_per_w = distance_m / world_dist if world_dist > 0 and distance_m and distance_m > 0 else 1.0

    if not pure_pixel_mode:
        ws_x, ws_y = pixel_to_world(H, scx, scy)
        wf_x, wf_y = pixel_to_world(H, fcx, fcy)
        world_dist = np.hypot(wf_x - ws_x, wf_y - ws_y)
        eff_dist = distance_m if distance_m else world_dist
        print(f"  [Geometry] {eff_dist}m = {world_dist:.2f} world units  (scale {scale_m_per_w:.4f} m/wu)")
        cx_check = (corners[0][0] + corners[2][0]) / 2.0
        cy_check = (corners[0][1] + corners[2][1]) / 2.0
        wx0, wy0 = pixel_to_world(H, cx_check,     cy_check)
        wx1, wy1 = pixel_to_world(H, cx_check + 1, cy_check)
        m_per_px  = np.hypot(wx1 - wx0, wy1 - wy0) * scale_m_per_w
        print(f"  [Geometry] Scale check: 1 px at field centre ≈ {m_per_px:.4f} m")

    print("Opening capture...")
    cap     = open_capture(source)
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    if not vid_fps or vid_fps <= 0:
        vid_fps = 30.0
        print("  [FPS] Stream did not report FPS -- defaulting to 30fps")
    print(f"FPS: {vid_fps:.2f}")

    writer = AsyncWriter(VIDEO_OUTPUT, vid_fps, DISPLAY_SIZE) if VIDEO_OUTPUT else None

    tracks          = {}
    finished_tracks = {}   # tracks removed from active set but kept for graphing
    id_remap        = {}   # bytetrack tid to canonical tid (for re-identified tracks)
    speed_logs    = defaultdict(list)
    start_events  = defaultdict(list)
    finish_events = defaultdict(list)

    def rand_color(tid):
        hue = (tid * 0.618033988749895) % 1.0
        hsv = np.uint8([[[int(hue * 179), 220, 210]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    frame_idx       = 0
    fps_display     = 0.0
    t_fps           = time.perf_counter()
    last_det_result = None
    paused          = False
    detection_active = True   # for realtime: toggled by on-screen button
    mouse_pos       = [None] # mutable container for mouse click coords

    def _on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_pos[0] = (x, y)

    if mode == "realtime":
        cv2.namedWindow("Speed Tracker", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Speed Tracker", _on_mouse)

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
                    pure_pixel_mode = (pixel_mode and distance_m is None)
                    scx = (start_seg[0][0] + start_seg[1][0]) / 2.0
                    scy = (start_seg[0][1] + start_seg[1][1]) / 2.0
                    fcx = (finish_seg[0][0] + finish_seg[1][0]) / 2.0
                    fcy = (finish_seg[0][1] + finish_seg[1][1]) / 2.0
                    if pure_pixel_mode:
                        H     = build_pixel_homography(1.0)
                        H_inv = np.linalg.inv(H)
                        scale_m_per_w = 1.0
                        print("  [Geometry] Recalibrated: pure pixel mode (px/s)")
                    elif pixel_mode:
                        px_dist = np.hypot(fcx - scx, fcy - scy)
                        if px_dist < 1: px_dist = 1.0
                        px_to_m = distance_m / px_dist
                        H       = build_pixel_homography(px_to_m)
                        H_inv   = np.linalg.inv(H)
                        ws_x, ws_y = pixel_to_world(H, scx, scy)
                        wf_x, wf_y = pixel_to_world(H, fcx, fcy)
                        world_dist = np.hypot(wf_x - ws_x, wf_y - ws_y)
                        scale_m_per_w = distance_m / world_dist if world_dist > 0 else 1.0
                        print(f"  [Geometry] Recalibrated: {distance_m}m = {world_dist:.2f} wu (scale {scale_m_per_w:.4f})")
                    else:
                        ws_x, ws_y = pixel_to_world(H, scx, scy)
                        wf_x, wf_y = pixel_to_world(H, fcx, fcy)
                        world_dist = np.hypot(wf_x - ws_x, wf_y - ws_y)
                        scale_m_per_w = distance_m / world_dist if world_dist > 0 else 1.0
                        print(f"  [Geometry] Recalibrated: {distance_m}m = {world_dist:.2f} wu (scale {scale_m_per_w:.4f})")
                    tracks.clear()
            paused = False

        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break

        frame = undistort_frame(frame, calib_map1, calib_map2)
        frame = cv2.resize(frame, DISPLAY_SIZE)
        frame_idx += 1
        t_now = (frame_idx / vid_fps) if mode == "offline" else time.perf_counter()

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
            cv2.imshow("Speed Tracker", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        FINISHED_TTL = 6.0   # seconds
        for _tid in [k for k, s in finished_tracks.items()
                     if t_now - s.last_seen_t > FINISHED_TTL]:
            del finished_tracks[_tid]

        annotated   = frame.copy()
        current_ids = set()

        det = last_det_result
        if det.boxes is not None and det.boxes.id is not None:
            boxes = det.boxes.xyxy.cpu().numpy()
            ids   = det.boxes.id.cpu().numpy().astype(int)
            confs = det.boxes.conf.cpu().numpy()

            # Pre-calculate active canonical IDs in this frame's detections 
            # so we don't attempt to re-ID against a track that is actually detected.
            current_frame_ctids = {id_remap.get(t, t) for t in ids}

            for box, tid, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box

                # skip noise box
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_BOX_AREA:
                    continue

                # use foot as position reference
                cx_px = (x1 + x2) * 0.5
                cy_px = float(y2)

                # FILTER ROI AFTER DETECTION
                if not point_in_polygon(int(cx_px), int(cy_px), corners):
                    continue

                ctid = id_remap.get(tid, tid)
                current_ids.add(ctid)

                if ctid in tracks and tracks[ctid].roi_exit_t is not None:
                    # Runner reappeared — cancel the exit and shift the
                    # entry time forward so the gap is not counted.
                    gap_duration = t_now - tracks[ctid].roi_exit_t
                    tracks[ctid].roi_entry_t += gap_duration
                    tracks[ctid].roi_exit_t   = None

                wx, wy = pixel_to_world(H, cx_px, cy_px)
                wx_m   = wx * scale_m_per_w
                wy_m   = wy * scale_m_per_w

                if tid not in tracks and ctid not in tracks:
                 
                    REID_DIST_M = 5.0
                    # REID_DIST_M = 100.0 if pure_pixel_mode else 5.0
                    best_old_tid  = None
                    best_old_dist = REID_DIST_M
                    best_is_finished = False

                    candidates = []
                    for old_tid, old_state in list(finished_tracks.items()):
                        candidates.append((old_tid, old_state, True))
                    # Also check currently "lost" tracks that haven't expired to finished_tracks yet
                    for old_tid, old_state in list(tracks.items()):
                        if old_tid not in current_frame_ctids:
                            candidates.append((old_tid, old_state, False))

                    for old_tid, old_state, is_finished in candidates:
                        # Project where this track would be NOW using its
                        # last known velocity from the Kalman state, so the search
                        # window moves with the runner rather than being anchored
                        # to the last-seen position.
                        dt_since = t_now - old_state.last_seen_t
                        pred_wx = float(old_state.kf.x[0].item()) + float(old_state.kf.x[2].item()) * dt_since
                        pred_wy = float(old_state.kf.x[1].item()) + float(old_state.kf.x[3].item()) * dt_since
                        d = np.hypot(wx_m - pred_wx, wy_m - pred_wy)
                        if d < best_old_dist:
                            best_old_dist = d
                            best_old_tid  = old_tid
                            best_is_finished = is_finished

                    if best_old_tid is not None:
                        canonical_tid = best_old_tid
                        id_remap[tid] = canonical_tid

                        if best_is_finished:
                            old_state = finished_tracks.pop(best_old_tid)
                        else:
                            old_state = tracks[best_old_tid]
                            
                        old_state.kf.x[0] = wx_m
                        old_state.kf.x[1] = wy_m
                        old_state.prev_t = t_now
                        old_state.prev_wm = (wx_m, wy_m)
                        old_state.last_seen_t = t_now
                        old_state.roi_exit_t = None
                        old_state.last_box = (x1, y1, x2, y2)

                        tracks[canonical_tid] = old_state

                        if tid in speed_logs and tid != canonical_tid:
                            speed_logs[canonical_tid].extend(speed_logs.pop(tid))
                            speed_logs[canonical_tid].sort(key=lambda r: r[0])

                        if tid in start_events and tid != canonical_tid:
                            start_events[canonical_tid].extend(start_events.pop(tid))

                        if tid in finish_events and tid != canonical_tid:
                            finish_events[canonical_tid].extend(finish_events.pop(tid))
                    else:
                        canonical_tid = ctid
                        tracks[canonical_tid] = TrackState(
                            kf=make_kalman(vid_fps, wx_m, wy_m),
                            t=t_now,
                            px=int(cx_px),
                            py=int(cy_px),
                            wx_m=wx_m,
                            wy_m=wy_m,
                            color=rand_color(canonical_tid),
                            box=(x1, y1, x2, y2)
                        )
                else:
                    canonical_tid = ctid

                state = tracks[canonical_tid]
                kf = state.kf

                q_scale = 1.0 + (1.0 - float(conf)) * 2.0
                kf.Q = np.diag([0.01, 0.01, 0.08 * q_scale, 0.08 * q_scale])

                kf.predict()
                kf.update([wx_m, wy_m])

                dt_frame = t_now - state.prev_t
                smoothed_wx = float(kf.x[0].item())
                smoothed_wy = float(kf.x[1].item())

                if dt_frame > 1e-6:
                    prev_wx_m, prev_wy_m = state.prev_wm
                    speed_ms = np.hypot(smoothed_wx - prev_wx_m, smoothed_wy - prev_wy_m) / dt_frame
                    if not pure_pixel_mode:
                        speed_ms = min(speed_ms, 7.5)  # cap at 27 km/h — headroom for fast runners
                else:
                    speed_ms = state.last_speed_ms

                state.speed_buf.append(speed_ms)
                disp_ms = float(np.median(state.speed_buf))
                disp_kh = disp_ms * 3.6

                if state.prev_t > 0:
                    state.total_dist += disp_ms * dt_frame

                state.last_speed_ms = speed_ms
                state.prev_t = t_now
                state.prev_wm = (smoothed_wx, smoothed_wy)
                state.last_seen_t = t_now
                state.last_box = (x1, y1, x2, y2)
                state.last_disp_kh = disp_kh

                new_px, new_py = int(cx_px), int(cy_px)
                ppx, ppy = state.prev_px

                if segments_cross((ppx, ppy), (new_px, new_py), start_seg[0], start_seg[1]):
                    state.start_crossed.append(t_now)
                    start_events[canonical_tid].append(t_now)

                if segments_cross((ppx, ppy), (new_px, new_py), finish_seg[0], finish_seg[1]):
                    state.finish_crossed.append(t_now)
                    finish_events[canonical_tid].append(t_now)

                state.prev_px = (new_px, new_py)

                if frame_idx % LOG_EVERY_N == 0:
                    speed_logs[canonical_tid].append((t_now, disp_ms, state.total_dist))
        for tid in list(tracks.keys()):
            state = tracks[tid]
            if t_now - state.last_seen_t > 3.0:  
                if state.roi_exit_t is None:
                    state.roi_exit_t = state.last_seen_t
                finished_tracks[tid] = state
                del tracks[tid]
                continue

            col     = state.color
            disp_kh = state.last_disp_kh

            draw_col = col
            if tid not in current_ids:
                # Only mark RoI exit after a short grace period (0.5 s).
                # This prevents a single dropped frame from flashing 'done'
                # and stops the timer from flickering on brief occlusions.
                gap = t_now - state.last_seen_t
                if state.roi_exit_t is None and gap > 0.5:
                    state.roi_exit_t = state.last_seen_t
         
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
                bw = old_x2 - old_x1
                bh = old_y2 - old_y1
                # Guard: skip drawing if box has no area (uninitialized / degenerate)
                if bw < 4 or bh < 4:
                    continue
                old_cx = old_x1 + bw * 0.5
                old_cy = float(old_y2)
                dx, dy  = new_cx - old_cx, new_cy - old_cy
                bx1 = old_x1 + dx; by1 = old_y1 + dy
                bx2 = old_x2 + dx; by2 = old_y2 + dy
                state.last_box = (bx1, by1, bx2, by2)
                draw_col = tuple(max(0, c - 60) for c in col)
                
                if t_now - state.last_seen_t < 1.0:
                    cv2.rectangle(annotated, (int(bx1), int(by1)), (int(bx2), int(by2)), draw_col, 2, cv2.LINE_4)
             
            else:
                bx1, by1, bx2, by2 = state.last_box
                bw = bx2 - bx1; bh = by2 - by1
                if bw < 4 or bh < 4:
                    continue
                draw_col = col
                cv2.rectangle(annotated, (int(bx1), int(by1)), (int(bx2), int(by2)), draw_col, 2)

            # Small ID badge clipped to top-left corner of the box only.
            # All speed/RoI data is shown in the top-right panel instead.
            if tid in current_ids or (t_now - state.last_seen_t < 1.0):
                id_txt = f"ID {tid}"
                (tw, th), _ = cv2.getTextSize(id_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                lx = max(int(bx1), 0)
                ly = max(int(by1) - 4, th + 4)
                cv2.rectangle(annotated, (lx, ly - th - 3), (lx + tw + 4, ly + 2),
                              (20, 20, 20), -1)
                cv2.putText(annotated, id_txt, (lx + 2, ly - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_col, 1, cv2.LINE_AA)

        # ------------------------------------------------------------------
        # Top-right info panel — speed and/or RoI per subject
        # ------------------------------------------------------------------
        panel_rows = []
        for _tid, _state in sorted(tracks.items()):
            _is_active = (_tid in current_ids)
            _spd_kh    = _state.last_disp_kh
            if _state.roi_exit_t is None:
                _roi_s   = t_now - _state.roi_entry_t
                _roi_txt = f"{_roi_s:.2f}s"
                _roi_col = (0, 230, 140)
            else:
                _roi_s   = _state.roi_exit_t - _state.roi_entry_t
                _roi_txt = f"{_roi_s:.2f}s done"
                _roi_col = (160, 160, 160)
            panel_rows.append((_tid, _state.color, _spd_kh, _roi_txt, _roi_col, _is_active))

        if panel_rows and (SHOW_SPEED_OVERLAY or SHOW_ROI_OVERLAY):
            PFONT  = cv2.FONT_HERSHEY_SIMPLEX
            PFS    = 0.52
            PFT    = 1
            PPAD   = 6
            ROW_H  = 22
            COL_ID_W  = 52
            COL_SPD_W = 84
            COL_ROI_W = 112

            col_widths = [COL_ID_W]
            if SHOW_SPEED_OVERLAY: col_widths.append(COL_SPD_W)
            if SHOW_ROI_OVERLAY:   col_widths.append(COL_ROI_W)
            total_w = sum(col_widths) + PPAD * 2
            total_h = ROW_H * (len(panel_rows) + 1) + PPAD * 2

            px0 = annotated.shape[1] - total_w - 10
            py0 = 40

            _ov = annotated.copy()
            cv2.rectangle(_ov, (px0 - PPAD, py0 - PPAD),
                          (px0 + total_w, py0 + total_h), (15, 15, 15), -1)
            cv2.addWeighted(_ov, 0.60, annotated, 0.40, 0, annotated)

            # Header
            hx = px0
            cv2.putText(annotated, "ID", (hx + 2, py0 + ROW_H - 6),
                        PFONT, PFS, (200, 200, 200), PFT, cv2.LINE_AA)
            hx += col_widths[0]; ci = 1
            if SHOW_SPEED_OVERLAY:
                cv2.putText(annotated, "Speed", (hx + 2, py0 + ROW_H - 6),
                            PFONT, PFS, (200, 200, 200), PFT, cv2.LINE_AA)
                hx += col_widths[ci]; ci += 1
            if SHOW_ROI_OVERLAY:
                cv2.putText(annotated, "RoI Time", (hx + 2, py0 + ROW_H - 6),
                            PFONT, PFS, (200, 200, 200), PFT, cv2.LINE_AA)
            div_y = py0 + ROW_H + 2
            cv2.line(annotated, (px0 - PPAD, div_y), (px0 + total_w, div_y),
                     (60, 60, 60), 1)

            for row_i, (_tid, _col, _spd_kh, _roi_txt, _roi_col, _is_active) in enumerate(panel_rows):
                ry = py0 + ROW_H * (row_i + 2) - 4
                rx = px0
                _tc = _col if _is_active else tuple(max(0, c - 80) for c in _col)

                cv2.circle(annotated, (rx + 7, ry - 5), 5, _tc, -1)
                cv2.putText(annotated, f"{_tid}", (rx + 17, ry),
                            PFONT, PFS, _tc, PFT, cv2.LINE_AA)
                rx += col_widths[0]; ci = 1

                if SHOW_SPEED_OVERLAY:
                    if pure_pixel_mode:
                        _ss = f"{_spd_kh / 3.6:.1f} px/s"
                    else:
                        _ss = f"{_spd_kh:.1f} km/h"
                    if _spd_kh < 10:   _sc = (100, 255, 100)
                    elif _spd_kh < 20: _sc = (50,  220, 255)
                    elif _spd_kh < 28: _sc = (0,   165, 255)
                    else:              _sc = (60,   60,  255)
                    if not _is_active: _sc = tuple(max(0, c - 80) for c in _sc)
                    cv2.putText(annotated, _ss, (rx + 2, ry),
                                PFONT, PFS, _sc, PFT, cv2.LINE_AA)
                    rx += col_widths[ci]; ci += 1

                if SHOW_ROI_OVERLAY:
                    _rc = _roi_col if _is_active else tuple(max(0, c - 80) for c in _roi_col)
                    cv2.putText(annotated, _roi_txt, (rx + 2, ry),
                                PFONT, PFS, _rc, PFT, cv2.LINE_AA)

        t_wall      = time.perf_counter()
        fps_display = 0.9*fps_display + 0.1/max(t_wall-t_fps, 1e-9)
        t_fps       = t_wall
        hint        = "ESC=stop  R=recalibrate" if mode == "realtime" else "ESC=stop"
        cv2.putText(annotated,
                    f"FPS:{fps_display:.1f}  Frame:{frame_idx}  [{device.upper()}]  {hint}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,255,0), 2, cv2.LINE_AA)

        # --- Start / Stop button (realtime only) ---
        if mode == "realtime":
            bx1, by1, bx2, by2 = 10, 50, 310, 95
            if detection_active:
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 0, 160), -1)
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.putText(annotated, "STOP DETECTION", (bx1+18, by2-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 160, 0), -1)
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(annotated, "START DETECTION", (bx1+12, by2-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            if mouse_pos[0] is not None:
                mx, my = mouse_pos[0]
                if bx1 <= mx <= bx2 and by1 <= my <= by2:
                    detection_active = not detection_active
                    print("  [Gate] Detection", "started." if detection_active else "stopped.")
                mouse_pos[0] = None

        # --- Global RoI clock — bottom-right, gated by SHOW_ROI_OVERLAY ---
        if SHOW_ROI_OVERLAY:
            _all_st  = list(tracks.values()) + list(finished_tracks.values())
            _entered = [s for s in _all_st if s.roi_entry_t is not None]
            _active  = [s for s in tracks.values() if s.roi_exit_t is None]
            if _entered:
                _g_start = min(s.roi_entry_t for s in _entered)
                if _active:
                    _g_elapsed  = t_now - _g_start
                    _clock_col  = (0, 255, 100)
                    _clock_lbl  = f"RoI Clock: {_g_elapsed:.2f}s"
                else:
                    _g_elapsed  = max(s.roi_exit_t for s in _entered) - _g_start
                    _clock_col  = (100, 200, 255)
                    _clock_lbl  = f"RoI Clock: {_g_elapsed:.2f}s  [all out]"
                (cw, ch), _ = cv2.getTextSize(_clock_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cx0 = annotated.shape[1] - cw - 20
                cy0 = annotated.shape[0] - 16
                _ov2 = annotated.copy()
                cv2.rectangle(_ov2, (cx0 - 8, cy0 - ch - 6), (cx0 + cw + 8, cy0 + 6),
                              (20, 20, 20), -1)
                cv2.addWeighted(_ov2, 0.60, annotated, 0.40, 0, annotated)
                cv2.putText(annotated, _clock_lbl, (cx0, cy0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, _clock_col, 2, cv2.LINE_AA)

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
        if pure_pixel_mode:
            print(f"  ID {tid:3d}  Peak:{max(spds):6.2f}px/s  "
                  f"Avg:{np.mean(spds):6.2f}px/s  "
                  f"Dist:{dist:.1f}px  Finishes:{len(finish_events[tid])}  Splits:[{t_str}]")
        else:
            print(f"  ID {tid:3d}  Peak:{max(spds)*3.6:6.2f}km/h  "
                  f"Avg:{np.mean(spds)*3.6:6.2f}km/h  "
                  f"Dist:{dist:.1f}m  Finishes:{len(finish_events[tid])}  Splits:[{t_str}]")

    # Find valid IDs to prompt for names
    _filt_mult = 1.0 if pure_pixel_mode else 3.6
    _min_spd   = 5.0 if pure_pixel_mode else MIN_GRAPH_SPEED_KMH
    _min_dst   = 10.0 if pure_pixel_mode else MIN_GRAPH_DIST_M
    valid_tids = []
    for tid, records in speed_logs.items():
        if len(records) >= MIN_GRAPH_FRAMES:
            speeds_conv = [r[1] * _filt_mult for r in records]
            dist = records[-1][2] if len(records[-1]) > 2 else 0.0
            if float(np.mean(speeds_conv)) >= _min_spd and dist >= _min_dst:
                valid_tids.append(tid)

    runner_names = {}
    if valid_tids:
        root3 = tk.Tk()
        root3.title("Enter Runner Names")
        root3.geometry("400x400")
        root3.resizable(True, True)
        tk.Label(root3, text="Enter runner names:", 
                 wraplength=380, font=("Arial", 11)).pack(pady=10)
        
        canvas = tk.Canvas(root3)
        scrollbar = ttk.Scrollbar(root3, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        entries = {}
        for tid_val in sorted(valid_tids):
            row_frame = tk.Frame(scrollable_frame)
            row_frame.pack(fill="x", padx=10, pady=5)
            tk.Label(row_frame, text=f"ID {tid_val}:", width=8, anchor="e").pack(side="left")
            entry = tk.Entry(row_frame, width=30)
            entry.pack(side="left", padx=5)
            entries[tid_val] = entry

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def save_names():
            for t, ent in entries.items():
                val = ent.get().strip()
                if val:
                    runner_names[t] = val
            root3.destroy()

        btn_frame = tk.Frame(root3)
        btn_frame.pack(fill="x", pady=10)
        tk.Button(btn_frame, text="Generate Graphs", command=save_names, width=20, bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack()
        
        root3.mainloop()

    generate_speed_graphs(speed_logs, start_events, finish_events,
                          GRAPH_OUTPUT_TIME, GRAPH_OUTPUT_DIST, vid_fps,
                          runner_names=runner_names, pure_pixel_mode=pure_pixel_mode)
    if VIDEO_OUTPUT and os.path.exists(VIDEO_OUTPUT):
        print(f"  Video -> {VIDEO_OUTPUT}")

if __name__ == "__main__":
    main()