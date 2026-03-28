import cv2
import numpy as np
import os
import json
import glob

CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE_MM  = 30
IMAGE_EXT       = "jpg"
IMAGES_DIR      = "./calib_images"
OUTPUT_DIR      = "./calib_output"


def collect_images():
    if not os.path.isdir(IMAGES_DIR):
        os.makedirs(IMAGES_DIR, exist_ok=True)
        print(f"\n📁 Created folder: {IMAGES_DIR}")
        print(f"   → Place your chessboard images (.{IMAGE_EXT}) inside it and re-run.\n")
        exit(0)

    pattern     = os.path.join(IMAGES_DIR, f"*.{IMAGE_EXT}")
    image_paths = sorted(glob.glob(pattern))

    if not image_paths:
        print(f"\n❌ No .{IMAGE_EXT} images found in: {IMAGES_DIR}")
        print(f"   → Add chessboard photos and re-run.\n")
        exit(0)

    return image_paths


def find_corners(image_paths):
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_points    = []
    img_points    = []
    valid_images  = []
    failed_images = []
    image_size    = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"\n📸 Processing {len(image_paths)} images...")
    print(f"   Chessboard : {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners")
    print(f"   Square size: {SQUARE_SIZE_MM}mm\n")

    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"  ⚠️  Could not read: {os.path.basename(path)}")
            failed_images.append(path)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            valid_images.append(path)
            print(f"  ✅ [{idx+1:02d}/{len(image_paths)}] {os.path.basename(path)}")
        else:
            failed_images.append(path)
            print(f"  ❌ [{idx+1:02d}/{len(image_paths)}] {os.path.basename(path)}  — corners not found")

    print(f"\n  Valid  : {len(valid_images)}")
    print(f"  Failed : {len(failed_images)}")

    if image_size is None:
        print("\n❌ No images could be read at all. Check your image files.")
        exit(1)

    return obj_points, img_points, valid_images, image_size


def run_calibration(obj_points, img_points, image_size):
    print("\n🔧 Running calibration...")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    errors = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        errors.append(err)

    mean_error = float(np.mean(errors))
    return K, dist, mean_error


def save_calibration(K, dist, image_size, mean_error, n_valid):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    npz_path = os.path.join(OUTPUT_DIR, "calibration.npz")
    np.savez(npz_path,
             camera_matrix=K,
             dist_coeffs=dist,
             image_size=np.array(image_size),
             mean_reprojection_error=np.array(mean_error))

    json_path = os.path.join(OUTPUT_DIR, "calibration.json")
    with open(json_path, "w") as f:
        json.dump({
            "camera_matrix"             : K.tolist(),
            "dist_coeffs"               : dist.tolist(),
            "image_size"                : list(image_size),
            "focal_length_px"           : {"fx": round(K[0,0], 4), "fy": round(K[1,1], 4)},
            "principal_point_px"        : {"cx": round(K[0,2], 4), "cy": round(K[1,2], 4)},
            "mean_reprojection_error_px": round(mean_error, 4),
            "valid_images_used"         : n_valid,
            "chessboard_size"           : list(CHESSBOARD_SIZE),
            "square_size_mm"            : SQUARE_SIZE_MM,
        }, f, indent=2)

    return npz_path, json_path


def save_undistort_maps(K, dist, image_size):
    # alpha=0: crops to valid pixels only — no black borders, no zoom artifacts
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, image_size, alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, image_size, cv2.CV_16SC2)

    maps_path = os.path.join(OUTPUT_DIR, "undistort_maps.npz")
    np.savez(maps_path,
             map1=map1,
             map2=map2,
             new_camera_matrix=new_K,
             roi=np.array(roi))

    return maps_path


def save_visual_check(valid_images, K, dist):
    if not valid_images:
        return None

    sample = cv2.imread(valid_images[0])
    h, w   = sample.shape[:2]

    # alpha=0 matches the undistort maps — clean full view, no zoom
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    undistorted = cv2.undistort(sample, K, dist, None, new_K)

    comparison = np.hstack([sample, undistorted])
    cv2.putText(comparison, "ORIGINAL",    (20, 40),    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv2.putText(comparison, "UNDISTORTED", (w + 20, 40),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    out_path = os.path.join(OUTPUT_DIR, "sample_undistorted.jpg")
    cv2.imwrite(out_path, comparison)
    return out_path


def print_summary(K, dist, mean_error, npz_path, json_path, maps_path, sample_path):
    if mean_error < 0.5:
        quality = "🟢 Excellent  (< 0.5px)"
    elif mean_error < 1.0:
        quality = "🟡 Acceptable (0.5–1.0px)"
    else:
        quality = "🔴 Poor       (> 1.0px) — retake images"

    print("\n" + "="*55)
    print("  CALIBRATION RESULTS")
    print("="*55)
    print(f"\n  Focal length     fx = {K[0,0]:.2f}px")
    print(f"                   fy = {K[1,1]:.2f}px")
    print(f"  Principal point  cx = {K[0,2]:.2f}px")
    print(f"                   cy = {K[1,2]:.2f}px")
    print(f"\n  Reprojection error : {mean_error:.4f}px")
    print(f"  Quality            : {quality}")
    print(f"\n  Output files:")
    print(f"    Calibration data   → {npz_path}")
    print(f"    Human readable     → {json_path}")
    print(f"    Undistort maps     → {maps_path}")
    if sample_path:
        print(f"    Visual check       → {sample_path}")
    print("="*55)

    if mean_error > 1.0:
        print("\n  ⚠️  Tips to improve:")
        print("     • Use 20+ images from varied angles and distances")
        print("     • Cover all corners of the frame, not just center")
        print("     • Avoid motion blur — use good lighting")
        print("     • Keep chessboard completely flat (no warping)")
        print("     • Make sure CHESSBOARD_SIZE matches your actual board\n")
    else:
        print("\n  ✅ Calibration complete. Use calibration.npz in your tracking pipeline.\n")


def main():
    print("=" * 55)
    print("  CAMERA CALIBRATION — Athlete Speed Tracking")
    print("=" * 55)
    print(f"\n  Images dir  : {IMAGES_DIR}")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print(f"  Board size  : {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners")
    print(f"  Square size : {SQUARE_SIZE_MM}mm")

    image_paths = collect_images()

    obj_pts, img_pts, valid_imgs, image_size = find_corners(image_paths)

    if len(valid_imgs) < 10:
        print(f"\n❌ Only {len(valid_imgs)} valid images detected.")
        print(f"   Need at least 10 for reliable calibration. Add more images and re-run.\n")
        return

    K, dist, mean_error = run_calibration(obj_pts, img_pts, image_size)

    npz_path, json_path = save_calibration(K, dist, image_size, mean_error, len(valid_imgs))

    maps_path = save_undistort_maps(K, dist, image_size)

    sample_path = save_visual_check(valid_imgs, K, dist)

    print_summary(K, dist, mean_error, npz_path, json_path, maps_path, sample_path)


if __name__ == "__main__":
    main()