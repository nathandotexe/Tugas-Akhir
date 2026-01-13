import cv2
import csv
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from filterpy.kalman import KalmanFilter

# ====================================================
# USER SETTINGS
# ====================================================
VIDEO_SOURCE = "testrun.mp4"
CSV_OUTPUT = "speed_log.csv"

FPS_VIDEO = 30  # sesuaikan dengan video asli

# ====================================================
# HOMOGRAPHY SETUP (CONTOH, GANTI SESUAI LAPANGAN)
# ====================================================
# Titik pada frame (pixel)
src_pts = np.array([
    [320, 180],   # kiri atas
    [960, 180],   # kanan atas
    [320, 540],   # kiri bawah
    [960, 540]    # kanan bawah
], dtype=np.float32)

# Titik nyata lapangan (meter)
dst_pts = np.array([
    [0.0, 0.0],
    [10.0, 0.0],
    [0.0, 20.0],
    [10.0, 20.0]
], dtype=np.float32)

H, _ = cv2.findHomography(src_pts, dst_pts)

def pixel_to_world(px, py):
    p = np.array([[[px, py]]], dtype=np.float32)
    w = cv2.perspectiveTransform(p, H)
    return w[0][0][0], w[0][0][1]

# ====================================================
# KALMAN FILTER
# ====================================================
def create_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0 / FPS_VIDEO

    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ])

    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    kf.P *= 50
    kf.R *= 5
    kf.Q *= 0.01
    return kf

# ====================================================
# TRACKER STATE
# ====================================================
kalman_filters = {}
prev_positions = {}
prev_times = {}

def rand_color():
    return tuple(map(int, np.random.randint(50, 255, 3)))

colors = defaultdict(rand_color)

# ====================================================
# CSV INIT
# ====================================================
with open(CSV_OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "track_id",
        "px",
        "py",
        "world_x",
        "world_y",
        "speed_mps"
    ])

# ====================================================
# MAIN
# ====================================================
def main():
    print("Loading YOLO model...")
    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    prev_frame_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml"
        )

        det = results[0]

        if det.boxes is not None and det.boxes.id is not None:
            boxes = det.boxes.xyxy.cpu().numpy()
            ids = det.boxes.id.cpu().numpy().astype(int)

            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Kalman init
                if tid not in kalman_filters:
                    kf = create_kalman()
                    kf.x[:2] = np.array([[cx], [cy]])
                    kalman_filters[tid] = kf
                    prev_positions[tid] = None
                    prev_times[tid] = None

                kf = kalman_filters[tid]
                kf.predict()
                kf.update([cx, cy])

                px = int(kf.x[0].item())
                py = int(kf.x[1].item())

                wx, wy = pixel_to_world(px, py)

                speed_mps = 0.0
                now = time.time()

                if prev_positions[tid] is not None:
                    pwx, pwy = prev_positions[tid]
                    dt = now - prev_times[tid]
                    if dt > 0:
                        dist = np.sqrt((wx - pwx)**2 + (wy - pwy)**2)
                        speed_mps = dist / dt

                prev_positions[tid] = (wx, wy)
                prev_times[tid] = now

                # CSV log
                with open(CSV_OUTPUT, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        now, tid, px, py, wx, wy, speed_mps
                    ])

                # DRAW
                color = colors[tid]
                cv2.circle(frame, (px, py), 5, color, -1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    frame,
                    f"ID {tid} | {speed_mps:.2f} m/s",
                    (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_frame_time)
        prev_frame_time = now

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Drone Speed Tracking", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
