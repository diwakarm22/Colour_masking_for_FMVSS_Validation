import numpy as np
import cv2 as cv

# Load the image
img = cv.imread("C:/Users/uig40328/Downloads/attachments/side6.JPG")

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Convert to float32 for cornerHarris
gray = np.float32(gray)

# Perform Harris corner detection
dst = cv.cornerHarris(gray, 2, 1, 0.01)

# Dilate the corner points to make them more visible
dst = cv.dilate(dst, None)

# Threshold for selecting strong corners
threshold = 0.12 * dst.max()

# Find coordinates of corners
corners = np.where(dst > threshold)

# Convert coordinates to format required by goodFeaturesToTrack
corners = np.transpose(np.vstack(corners))

# Group corners into grids (assuming an 8x8 checkerboard)
grid_size = 8
if corners is not None and len(corners) >= grid_size * grid_size:
    corners = corners.reshape(-1, 2)
    num_grids = (grid_size - 1) * (grid_size - 1)
    grid_distances = np.zeros((num_grids, 4))
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            grid_index = i * (grid_size - 1) + j
            grid_corners = corners[i * grid_size + j:i * grid_size + j + 4]
            if len(grid_corners) == 4:
                for k in range(4):
                    for l in range(k + 1, 4):
                        distance = np.linalg.norm(grid_corners[k] - grid_corners[l])
                        grid_distances[grid_index, k] = distance

    # Estimate the minimum distance between corners based on distances between non-detected corners in the grid
    estimated_distance_between_non_detected_corners = 8 # Adjust as needed
    estimated_min_distance = estimated_distance_between_non_detected_corners

    # Use goodFeaturesToTrack with the estimated minimum distance
    corners = cv.goodFeaturesToTrack(gray, maxCorners=81, qualityLevel=0.1, minDistance=estimated_min_distance)

# Display the result
if corners is not None:
    img_with_corners = img.copy()
    corners = np.intp(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv.circle(img_with_corners, (x, y), 2, (0, 0, 255), -1)
    cv.imshow('Corners Detected', img_with_corners)

# Wait for user input to close the window
if cv.waitKey(0) & 0xFF == 27:
    cv.destroyAllWindows()
