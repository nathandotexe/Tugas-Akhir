import cv2
import numpy as np

VIDEO_PATH   = "walkrun.mp4"
CALIB_PATH   = "calib.npz"
DISPLAY_SIZE = (1280, 720)

clicked_pts = []


def mouse_event(event, x, y, flags, param):
    global clicked_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pts.append([x, y])
        print(f"Point {len(clicked_pts)}: ({x}, {y})")


# ---------------------------
# Calibration helpers
# ---------------------------
def load_calibration(path):
    data = np.load(path)
    K = data["camera_matrix"]
    dist = data["dist_coeffs"]
    calib_size = tuple(data["image_size"])  # (w, h)
    return K, dist, calib_size


def scale_camera_matrix(K, old_size, new_size):
    scale_x = new_size[0] / old_size[0]
    scale_y = new_size[1] / old_size[1]

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 2] *= scale_y

    return K_scaled


def undistort_frame(frame, K, dist):
    h, w = frame.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, K, dist, None, new_K)


# ---------------------------
# Main
# ---------------------------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error loading video.")
        return

    print(f"Original frame size: {frame.shape[1]}x{frame.shape[0]}")

    # Load calibration
    K, dist, calib_size = load_calibration(CALIB_PATH)

    # Match resolution if needed
    frame_h, frame_w = frame.shape[:2]
    if (frame_w, frame_h) != calib_size:
        print("Scaling camera matrix to match video resolution...")
        K = scale_camera_matrix(K, calib_size, (frame_w, frame_h))

    # Undistort
    frame = undistort_frame(frame, K, dist)

    h, w = frame.shape[:2]
    print(f"Undistorted frame size: {w}x{h}")

    # Resize ONLY for display
    scale_x = DISPLAY_SIZE[0] / w
    scale_y = DISPLAY_SIZE[1] / h

    display_frame = cv2.resize(frame, DISPLAY_SIZE)
    clone = display_frame.copy()

    cv2.namedWindow("Select 4 Ground Points", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select 4 Ground Points", *DISPLAY_SIZE)
    cv2.setMouseCallback("Select 4 Ground Points", mouse_event)

    print("\nClick 4 points on the ground plane in order.")
    print("Press Q to cancel.\n")

    while True:
        disp = clone.copy()

        for i, p in enumerate(clicked_pts):
            cv2.circle(disp, tuple(p), 6, (0, 0, 255), -1)
            cv2.putText(disp, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if len(clicked_pts) > 1:
            for i in range(1, len(clicked_pts)):
                cv2.line(disp, tuple(clicked_pts[i - 1]), tuple(clicked_pts[i]),
                         (0, 255, 180), 1)
            if len(clicked_pts) == 4:
                cv2.line(disp, tuple(clicked_pts[3]), tuple(clicked_pts[0]),
                         (0, 255, 180), 1)

        cv2.putText(disp, f"Points: {len(clicked_pts)}/4  |  Q=cancel  BACKSPACE=undo",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Select 4 Ground Points", disp)

        if len(clicked_pts) == 4:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Cancelled.")
            cv2.destroyAllWindows()
            return
        if key == 8 and clicked_pts:
            removed = clicked_pts.pop()
            print(f"Removed point: {removed}")

    cv2.destroyAllWindows()

    # Convert display coords → original undistorted coords
    pix = []
    for p in clicked_pts:
        x = p[0] / scale_x
        y = p[1] / scale_y
        pix.append([x, y])

    pix = np.array(pix, dtype=np.float32)

    print("\nEnter the real-world coordinates (meters) for each point.")
    print("Origin (0, 0) is wherever you set Point 1.\n")

    world_pts = []
    for i in range(4):
        X = float(input(f"  Point {i+1} world X (meters): "))
        Y = float(input(f"  Point {i+1} world Y (meters): "))
        world_pts.append([X, Y])

    world = np.array(world_pts, dtype=np.float32)

    # Compute homography
    H, mask = cv2.findHomography(pix, world, cv2.RANSAC, 5.0)

    if H is None:
        print("Homography computation failed — check your points.")
        return

    inliers = int(mask.sum()) if mask is not None else 4

    print(f"\nHomography matrix:\n{H}")
    print(f"Inliers: {inliers}/4")

    # Save everything
    np.save("H.npy", H)
    np.save("pixel_points.npy", pix)
    np.save("world_points.npy", world)

    print("\nSaved: H.npy, pixel_points.npy, world_points.npy")

    _verify_reprojection(H, pix, world)


# ---------------------------
# Verification
# ---------------------------
def _verify_reprojection(H, pix, world):
    print("\nReprojection check (pixel → world):")

    errors = []
    for i, (px, wpt) in enumerate(zip(pix, world)):
        projected = cv2.perspectiveTransform(px.reshape(1, 1, 2), H)[0][0]
        err = np.linalg.norm(projected - wpt)
        errors.append(err)

        print(f"  Point {i+1}: pixel {px} → projected ({projected[0]:.4f}, {projected[1]:.4f}) "
              f"| expected ({wpt[0]:.4f}, {wpt[1]:.4f}) | error {err:.4f}m")

    mean_err = np.mean(errors)
    print(f"\n  Mean reprojection error: {mean_err:.4f}m")

    if mean_err < 0.05:
        print("  Quality: good")
    elif mean_err < 0.15:
        print("  Quality: acceptable — consider repicking points")
    else:
        print("  Quality: poor — redo everything more carefully")


if __name__ == "__main__":
    main()