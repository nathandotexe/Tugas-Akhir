import cv2
import numpy as np
import os

VIDEO_PATH = "walkrun.mp4"
CALIB_PATH = "calibration_output/calibration.npz"

clicked_pts = []


def load_calibration(calib_path):
    if calib_path is None or not os.path.exists(calib_path):
        print(f"[Calib] Not found: {calib_path} — skipping undistortion.")
        return None, None

    data = np.load(calib_path)

    if "map1" in data and "map2" in data:
        print(f"[Calib] Loaded undistort maps from {calib_path}")
        return data["map1"], data["map2"]

    if "camera_matrix" in data and "dist_coeffs" in data:
        K    = data["camera_matrix"]
        dist = data["dist_coeffs"]
        w, h = data["image_size"].astype(int)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_16SC2)
        print(f"[Calib] Computed undistort maps from {calib_path}")
        return map1, map2

    print("[Calib] Unrecognised .npz format — skipping undistortion.")
    return None, None


def undistort_frame(frame, map1, map2):
    if map1 is None or map2 is None:
        return frame
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


def mouse_event(event, x, y, flags, param):
    global clicked_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pts.append([x, y])
        print(f"Point {len(clicked_pts)}: ({x}, {y})")


def main():
    map1, map2 = load_calibration(CALIB_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error loading video.")
        return

    frame = undistort_frame(frame, map1, map2)
    clone = frame.copy()

    cv2.namedWindow("Select 4 Ground Points")
    cv2.setMouseCallback("Select 4 Ground Points", mouse_event)

    print("Click 4 points on the ground plane in order.")
    print("Recommended: corners of a known rectangle on the track (e.g. lane line intersections).")
    print("Press Q to cancel.\n")

    while True:
        disp = clone.copy()
        for i, p in enumerate(clicked_pts):
            cv2.circle(disp, tuple(p), 6, (0, 0, 255), -1)
            cv2.putText(disp, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if len(clicked_pts) > 1:
            for i in range(1, len(clicked_pts)):
                cv2.line(disp, tuple(clicked_pts[i-1]), tuple(clicked_pts[i]), (0, 255, 180), 1)
            if len(clicked_pts) == 4:
                cv2.line(disp, tuple(clicked_pts[3]), tuple(clicked_pts[0]), (0, 255, 180), 1)

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

    pix = np.array(clicked_pts, dtype=np.float32)

    print("\nEnter the real-world coordinates (meters) for each point.")
    print("Origin (0, 0) is wherever you set Point 1.\n")
    world_pts = []
    for i in range(4):
        X = float(input(f"  Point {i+1} world X (meters): "))
        Y = float(input(f"  Point {i+1} world Y (meters): "))
        world_pts.append([X, Y])

    world = np.array(world_pts, dtype=np.float32)

    H, mask = cv2.findHomography(pix, world, cv2.RANSAC, 5.0)

    if H is None:
        print("Homography computation failed — check your points.")
        return

    inliers = int(mask.sum()) if mask is not None else 4
    print(f"\nHomography matrix:\n{H}")
    print(f"Inliers: {inliers}/4")

    np.save("H.npy", H)
    np.save("pixel_points.npy", pix)
    np.save("world_points.npy", world)

    print("\nSaved: H.npy, pixel_points.npy, world_points.npy")

    _verify_reprojection(H, pix, world)


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
        print("  Quality: acceptable — consider repicking points more precisely")
    else:
        print("  Quality: poor — re-run and pick more accurate ground points")


if __name__ == "__main__":
    main()