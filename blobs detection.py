import numpy as np
import cv2 as cv
import os

def compute_homography(image_path):
    ' Harris corner detection to detect all corners'
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, maxCorners=81, qualityLevel=0.01, minDistance=10)
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                              criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    ' Selecting the 4 corners '
    if corners is not None:
        img_with_corners = img.copy()

        for corner in corners:
            x, y = corner.ravel()
            cv.circle(img_with_corners, (int(x), int(y)), 2, (0, 0, 255), -1)

        sorted_corners = sorted(corners, key=lambda corner: (corner[0][0], corner[0][1]))
        selected_corners = [sorted_corners[10], sorted_corners[-11], sorted_corners[16], sorted_corners[-17]]
        selected_corner_pixels = [(corner.ravel()[0], corner.ravel()[1]) for corner in selected_corners]

        for corner in selected_corner_pixels:
            x, y = corner
            cv.circle(img_with_corners, (int(x), int(y)), 3, (0, 255, 0), -1)

        cv.imshow("Corners Detected", img_with_corners)
        cv.waitKey(0)
        cv.destroyAllWindows()

        ' XYZ of the selected corners, z is zero since the checkerboard is on a planar surface'
        real_world_coords = np.array([[-1.5, -1.5], [1.5, -1.5], [-1.5, 1.5], [1.5, 1.5]], dtype=np.float32)

        'Pixel UV values of the corner in the image'
        src_pts = np.array(selected_corner_pixels, dtype=np.float32)
        print(f"source points:\n",src_pts)
        print("dst points:\n",real_world_coords)

        dst_pts = real_world_coords

        ' Compute homography matrix using RANSAC'
        homography_matrix, _ = cv.findHomography(src_pts, dst_pts, method=cv.RANSAC)

        return homography_matrix
    else:
        return None

def decompose_homography(H, K):


    return rvec, t

def main():
    'the intrinsic matrix K'
    fx, fy = .34511988, 1663.333
    cx, cy = 0,0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    k1 = 5
    k2 = 6
    k3 = 7
    p1 = 8
    p2 = 8

    folder_path = "C:/Users/uig40328/Downloads/attachments"
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing image: {image_path}")
            homography_matrix = compute_homography(image_path)
            if homography_matrix is not None:
                print(f"Computed homography matrix for {filename}:")
                print(homography_matrix)

                # Decompose the homography matrix into rotation matrix R and translation vector t
                rvec , t = decompose_homography(homography_matrix, K)

                print(f"Rotation vector:\n",rvec )
                print("Translation Vector:\n", t)
            else:
                print(f"No homography matrix computed for {filename}")

if __name__ == "__main__":
    main()
