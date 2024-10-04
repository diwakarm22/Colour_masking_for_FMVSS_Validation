import numpy as np
import cv2 as cv
import os
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
CHECKERBOARD = (7,7)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('C:/Users/uig40328/Downloads/blenderout/*.png')
for fname in images:
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)
N_OK = len(objpoints)
K = np.zeros((3, 3))

_, _, _, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints, imgpoints, gray.shape[::-1], None, distCoeffs
)
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")
print(objpoints)
print(objpoints[0].shape)

print(imgpoints)
print(rvecs)
print(tvecs)
'''
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
    'Compute the inverse of the intrinsic matrix'
    K_inv = np.linalg.inv(K)

    # Compute the scaling factor
    lambda_ = 1 / np.linalg.norm(np.dot(K_inv, H[:, 0]))

    # Compute the rotation matrix
    r1 = lambda_ * np.dot(K_inv, H[:, 0])
    r2 = lambda_ * np.dot(K_inv, H[:, 1])
    r3 = np.cross(r1, r2)
    R = np.column_stack((r1, r2, r3))
    rvec,_  = cv.Rodrigues(R)

    # Compute the translation vector
    t = lambda_ * np.dot(K_inv, H[:, 2])

    return rvec, t

def main():
    'the intrinsic matrix K'
    fx, fy = .34511988, 1663.333
    cx, cy = 0,0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

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
    main() '''
