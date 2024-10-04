import numpy as np
import cv2 as cv
import os

# plane 0.02
def corner_detection(image_path):
    img = cv.imread(image_path)
    roi = img[147:800, 300:930]
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, maxCorners=81, qualityLevel=0.01, minDistance=8)
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                              criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    ' Selecting the 4 corners '
    if corners is not None:
        img_with_corners = roi.copy()

        for corner in corners:
            x, y = corner.ravel()
            cv.circle(img_with_corners, (int(x), int(y)), 2, (0, 0, 255), -1)

        sorted_corners = sorted(corners, key=lambda corner: (corner[0][0], corner[0][1]))
        selected_corners = [sorted_corners[9], sorted_corners[-11], sorted_corners[10], sorted_corners[-9]]
        selected_corner_pixels = [(corner.ravel()[0], corner.ravel()[1]) for corner in selected_corners]

        for corner in selected_corner_pixels:
            x, y = corner
            cv.circle(img_with_corners, (int(x), int(y)), 2, (0, 255, 0), -1)

        cv.imshow("Corners Detected", img_with_corners)
        cv.waitKey(0)
        cv.destroyAllWindows()

        ' XYZ of the selected corners, z is zero since the checkerboard is on a planar surface'
        real_world_coords = np.array([[-1.5, -1.5,0], [1.5, -1.5,0], [-1.5, 1.5,0], [1.5, 1.5,0]], dtype=np.float32)

        'Pixel UV values of the corner in the image'
        src_pts = np.array(selected_corner_pixels, dtype=np.float32)
        dst_pts = real_world_coords

        return dst_pts, src_pts


def calibration(dc, dst_pts, src_pts, K):
    ret, rvec, tvec = cv.solvePnP(dst_pts, src_pts, K, dc)
    # rmat, _ = cv.Rodrigues(rvec)
    return ret, rvec, tvec


def main():
    'the intrinsic matrix K'
    fx, fy = .34511988, 1663.333
    cx, cy = 0, 0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    k1 = 344
    k2 = -9.73
    k3 = 43
    p1 = -50
    p2 = 34.4
    dc = np.array([k1, k2, p1, p2, k3]).reshape(-1, 1)

    folder_path = "C:/Users/uig40328/Downloads/blenderout"
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing image: {image_path}")
            dst_pts, src_pts = corner_detection(image_path)
            ret, rvec, tvec = calibration(dc, dst_pts, src_pts, K)
            print(f"Rotation vector:\n", rvec)
            print("Translation Vector:\n", tvec)



if __name__ == "__main__":
    main()
