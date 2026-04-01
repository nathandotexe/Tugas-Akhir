import cv2
import glob
import os
import time
import numpy as np

CHESSBOARD   = (8, 5)
SQUARE_SIZE  = 0.03
IMAGES_GLOB  = "calib_images/*.jpg"
OUT_FILE     = "calib.npz"
DETECT_SCALE = 0.5
SAVE_DEBUG   = True
DEBUG_DIR    = "calib_debug"

FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

def make_objp():
    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp

def detect(gray_full):
    h, w = gray_full.shape
    small = cv2.resize(gray_full, (int(w * DETECT_SCALE), int(h * DETECT_SCALE)),
                       interpolation=cv2.INTER_AREA)
    ok, corners = cv2.findChessboardCorners(small, CHESSBOARD, FLAGS)
    if not ok:
        return False, None
    corners_full = corners / DETECT_SCALE
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)
    corners_refined = cv2.cornerSubPix(gray_full, corners_full, (11, 11), (-1, -1), crit)
    return True, corners_refined

def try_imshow(title, img):
    try:
        cv2.imshow(title, img)
        cv2.waitKey(200)
    except Exception:
        pass

def main():
    images = sorted(glob.glob(IMAGES_GLOB))
    if not images:
        print(f"No images found at: {IMAGES_GLOB}")
        return

    if SAVE_DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    objp = make_objp()
    objpoints, imgpoints, used = [], [], []
    t0 = time.time()
    gray = None

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"[{i+1}/{len(images)}] Failed to load: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = detect(gray)
        name = os.path.basename(fname)

        if found:
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            used.append(fname)
            print(f"[{i+1}/{len(images)}] FOUND:     {name}")

            if SAVE_DEBUG:
                dbg = img.copy()
                cv2.drawChessboardCorners(dbg, CHESSBOARD, corners, True)
                cv2.imwrite(os.path.join(DEBUG_DIR, name), dbg)

            try_imshow("preview", img)
        else:
            print(f"[{i+1}/{len(images)}] not found: {name}")

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    if not objpoints:
        print("No corners found in any image.")
        return

    if gray is None:
        print("No images could be loaded.")
        return

    img_size = (gray.shape[1], gray.shape[0])
    print(f"\nCalibrating with {len(objpoints)}/{len(images)} images...")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    per_err = []
    for j in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[j], rvecs[j], tvecs[j], K, dist)
        per_err.append(cv2.norm(imgpoints[j], proj, cv2.NORM_L2) / len(proj))

    print(f"\nRMS error  : {rms:.4f} px")
    print(f"Mean/std   : {np.mean(per_err):.4f} / {np.std(per_err):.4f} px")
    print(f"K:\n{K}")
    print(f"dist: {dist.ravel()}")

    if SAVE_DEBUG:
        print(f"\nDebug images saved to: {DEBUG_DIR}/")

    np.savez(OUT_FILE, camera_matrix=K, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs,
             image_size=img_size, image_files=np.array(used), rms=rms)
    print(f"Saved: {OUT_FILE}  ({len(used)} images, {time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()