import cv2
import time
from ultralytics import YOLO

# -----------------------------
# User Settings
# -----------------------------
MODEL_PATH = "best.pt"          # your YOLOv8 model
VIDEO_PATH = "testrun.mp4"      # your input video
OUT_PATH   = "output.mp4"       # output video path
CONF_THRES = 0.5                # confidence threshold

# -----------------------------
# Load Model
# -----------------------------
model = YOLO(MODEL_PATH)
print("[+] Model loaded successfully")

# -----------------------------
# Load Video
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error loading video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int (cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int (cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUT_PATH, fourcc, fps, (width, height))

print(f"[+] Video resolution: {width}x{height} @ {fps} FPS")
print("[+] Processing video...\n")

prev_time = time.time()

# -----------------------------
# Process Frame-by-Frame
# -----------------------------
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 320))
    if not ret:
        break

    # FPS calculation
    current_time = time.time()
    fps_live = 1 / (current_time - prev_time)
    prev_time = current_time

    # YOLOv8 inference
    results = model.predict(
        frame,
        conf=CONF_THRES,
        verbose=False
    )

    # parse results
    detections = results[0].boxes

    if detections is not None:
        for det in detections:
            cls = int(det.cls[0])
            conf = float(det.conf[0])

            # Only detect PERSON (class 0 in COCO)
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, det.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"Person {conf:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

    # Draw FPS
    cv2.putText(frame,
                f"FPS: {fps_live:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    out.write(frame)
    cv2.imshow("Detection Preview", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("[+] Done. Saved to:", OUT_PATH)
