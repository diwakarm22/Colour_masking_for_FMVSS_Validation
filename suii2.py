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
origin_offset_x = 0.25
origin_offset_y = 0.25

# Calculate the spacing between corners
x_spacing = 0.25
y_spacing = 0.25

# Initialize an empty array to store corner positions
arr_3d = np.zeros((num_rows * num_cols, 3), dtype=np.float64)

# Fill the array with corner positions
for i in range(num_rows):
    for j in range(num_cols):
        index = i * num_cols + j
        x = origin_offset_x + i * x_spacing
        y = origin_offset_y + j * y_spacing
        arr_3d[index] = [x, y, 0]

# Reshape the array to include an extra dimension
corner_positions_3D = arr_3d.reshape(1, 49, 3)

# Extracting path of individual image stored in a given directory
images = glob.glob('C:/Users/uig40328/Downloads/blenderout/*.png')

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
        objpoints.append(corner_positions_3D)
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

# Convert the distortion coefficients to the correct shape
k1 = 345.7533
k2 = -11.559854
k3 = 44.552719
k4 = -51.553291

D = np.array([k1, k2, k3, k4], dtype=np.float64).reshape(-1, 1)

# Convert the lists of points to arrays
objpoints = np.array(objpoints)
imgpoints = np.array(imgpoints)

# Perform camera calibration
rvecs_degrees = np.array([57,0,-90])
rvecs = [np.radians(rvecs_degrees)]
tvecs = np.array([ 3.826, 0.0066,1.0163])
print(rvecs)


# Use the obtained rvecs and tvecs for further processing