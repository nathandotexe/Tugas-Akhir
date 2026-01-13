import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# ============================
# USER CONFIG
# ============================
VIDEO_PATH = "testrun.mp4"
MODEL_PATH = "yolov8s.pt"   # atau best.pt hasil training kamu
CONF_THRES = 0.5

# ============================
# LOAD MODEL
# ============================
model = YOLO(MODEL_PATH)
print("[+] YOLO model loaded")

# ============================
# VIDEO INPUT
# ============================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Cannot open video")

# ============================
# TRACK COLOR PER ID
# ============================
def random_color():
    return tuple(np.random.randint(50, 255, 3).tolist())

track_colors = defaultdict(random_color)

# ============================
# MAIN LOOP
# ============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    # ============================
    # YOLOv8 + ByteTrack
    # ============================
    results = model.track(
        source=frame,
        persist=True,
        conf=CONF_THRES,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, classes):
            # hanya pelari (person = class 0 COCO)
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            color = track_colors[track_id]

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Centroid
            cv2.circle(frame, (cx, cy), 4, color, -1)

            # ID label
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow("Runner Tracking - ByteTrack", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
