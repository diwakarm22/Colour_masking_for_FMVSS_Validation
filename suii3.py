import cv2
import numpy as np
import os
import glob

# Define the dimensions of the checkerboard
CHECKERBOARD = (7, 7)
num_rows = 7
num_cols = 7
width = 2  # meters
height = 2  # meters
origin_offset_x = 6.25
origin_offset_y = 0.75

# Calculate the spacing between corners
x_spacing =   0.25
y_spacing = - 0.25

# Initialize an empty array to store corner positions
arr_3d = np.zeros((num_rows * num_cols, 3), dtype=np.float64)

# Fill the array with corner positions
for i in range(num_rows):
    for j in range(num_cols):
        index = i * num_cols + j
        x = origin_offset_x + j * x_spacing
        y = origin_offset_y + i * y_spacing
        arr_3d[index] = [x, y, 0]

# Reshape the array to include an extra dimension
corner_positions_3D = arr_3d.reshape(1, 49, 3)

# Extracting path of individual image stored in a given directory
images = glob.glob('C:/Users/uig40328/Downloads/blendout2/*.png')

objpoints = []  # 3D point in real world space
imgpoints = []  # 2D points in image plane.

for filename in images:
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If desired number of corners can be detected
    if ret == True:
        # Refine the pixel coordinates of corners
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Append the refined corners to the list
        objpoints.append(corner_positions_3D[0])
        imgpoints.append(corners)

        # Draw and label the corners
        for i, corner in enumerate(corners):
            x, y = corner.ravel()
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the image with labeled corners
        cv2.imshow('Labeled Corners', image)
        cv2.waitKey(0)

cv2.destroyAllWindows()

# Convert the lists of points to arrays
objpoints = np.array([objpoints])
imgpoints = np.array(imgpoints)

k1 = 0.03037981634812789
k2 = 0.03318714459852845
k3 = -0.3596519446149046
k4 = 0.6834454498875479

D = np.array([k1, k2, k3, k4], dtype=np.float64).reshape(-1, 1)
cameraMatrix = np.array([[346.0287440066604, 0.0, 613.0398470717112], [0.0, 348.8662016762894, 486.47324759020375], [0.0, 0.0, 1.0]])

# Perform camera calibration
ret  ,_, rvecs,tvecs ,_ = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1],cameraMatrix, D)

#retval, rvec, tvec = cv2.solvePnP (objpoints, imgpoints, cameraMatrix, D)
print(tvecs)