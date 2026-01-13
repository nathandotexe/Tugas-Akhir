import cv2
import numpy as np

VIDEO_PATH = "testrun.mp4"

clicked_pts = []

def mouse_event(event, x, y, flags, param):
    global clicked_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pts.append([x, y])
        print(f"Point {len(clicked_pts)}: {x}, {y}")

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 600))
    cap.release()

    if not ret:
        print("Error loading video.")
        return

    clone = frame.copy()
    cv2.namedWindow("Select 4 Ground Points")
    cv2.setMouseCallback("Select 4 Ground Points", mouse_event)

    print("Click 4 points on the ground plane in order.")
    print("Recommended: corners of a rectangle on the field.")

    while True:
        disp = clone.copy()
        for p in clicked_pts:
            cv2.circle(disp, tuple(p), 5, (0, 0, 255), -1)

        cv2.imshow("Select 4 Ground Points", disp)

        if len(clicked_pts) == 4:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    cv2.destroyAllWindows()

    pix = np.array(clicked_pts, dtype=np.float32)

    # Ask user for world coordinates of those 4 points
    print("\nEnter the real-world coordinates (meters) for each point:")
    world_pts = []
    for i in range(4):
        X = float(input(f"Point {i+1} world X (meters): "))
        Y = float(input(f"Point {i+1} world Y (meters): "))
        world_pts.append([X, Y])

    world = np.array(world_pts, dtype=np.float32)

    # Compute homography
    H, _ = cv2.findHomography(pix, world)

    print("\nHomography matrix computed:")
    print(H)

    np.save("H.npy", H)
    np.save("pixel_points.npy", pix)
    np.save("world_points.npy", world)

    print("\nSaved: H.npy, pixel_points.npy, world_points.npy")

if __name__ == "__main__":
    main()
