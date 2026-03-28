import cv2
import numpy as np
from ultralytics import YOLO
import torch

# ======================================================
# USER CONFIG
# ======================================================
MODEL_PATH = "best.pt"
VIDEO_PATH = "testrun.mp4"
OUTPUT_PATH = "output_markers.mp4"

TARGET_W, TARGET_H = 640, 360
FPS_SAFETY = 30

# =============================
# HOMOGRAPHY SETUP
# =============================
# 👉 Fill these with your own values
PIX_POINTS = np.array([
    # [x, y] pixel coords of 4 known ground points
], dtype=np.float32)

WORLD_POINTS = np.array([
    # [X, Y] real coordinates in meters
], dtype=np.float32)

if len(PIX_POINTS) == 4:
    H, _ = cv2.findHomography(PIX_POINTS, WORLD_POINTS)
else:
    H = None
    print("⚠ Homography not set. Using fallback (GSD).")


# Fallback (if homography missing)
GSD = 0.02  # meters per pixel


# ======================================================
# LOAD YOLO MODEL
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
model.to(device)

# ======================================================
# VIDEO SETUP
# ======================================================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or FPS_SAFETY

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (TARGET_W, TARGET_H))

prev_world_x = None
prev_t = None
speed_mps = 0.0

# ======================================================
# FUNCTIONS
# ======================================================
def pixel_to_meter(x, y):
    """Convert pixel point to world coordinates using homography."""
    if H is None:
        return x * GSD, y * GSD

    p = np.array([x, y, 1.0])
    P = H @ p
    P /= P[2]  # normalize
    return P[0], P[1]


# ======================================================
# PROCESS VIDEO
# ======================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (TARGET_W, TARGET_H))
    Hf, Wf, _ = frame.shape

    # Run YOLO on GPU
    results = model.predict(frame, conf=0.5, verbose=False, device=device)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Track the *bottom center* of the bbox
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            # Convert bottom center pixel → meters
            world_x, world_y = pixel_to_meter(cx, cy)

            # Speed estimation
            now = cv2.getTickCount()
            if prev_world_x is not None:
                dt = (now - prev_t) / cv2.getTickFrequency()

                if dt > 0:
                    dx = world_x - prev_world_x
                    speed_mps = dx / dt

            prev_world_x = world_x
            prev_t = now

    # Speed UI
    cv2.rectangle(frame, (10, 10), (260, 70), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Speed: {speed_mps:.2f} m/s",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    out.write(frame)
    cv2.imshow("Speed Estimator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
