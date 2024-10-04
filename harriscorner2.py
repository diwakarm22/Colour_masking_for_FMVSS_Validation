import numpy as np
import cv2 as cv

# Load the image
img = cv.imread("C:/Users/uig40328/Downloads/attachments/dark3.JPG")

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Convert to float32 for cornerHarris
gray = np.float32(gray)

# Perform Harris corner detection
dst = cv.cornerHarris(gray, 2, 1, 0.01)

# Dilate the corner points to make them more visible
dst = cv.dilate(dst, None)

# Threshold for selecting strong corners
threshold = 0.12* dst.max()

# Find coordinates of corners
corners = np.where(dst > threshold)

# Convert coordinates to format required by goodFeaturesToTrack
corners = np.transpose(np.vstack(corners))

# Use goodFeaturesToTrack to detect multiple corners
corners = cv.goodFeaturesToTrack(gray, maxCorners=90, qualityLevel=0.1, minDistance=8,)

# Draw circles around the corners
if corners is not None:
    corners = np.intp(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv.circle(img, (x, y), 2, (0, 0, 255), -1)

# Display the result
cv.imshow('Corners Detected', img)

# Wait for user input to close the window
if cv.waitKey(0) & 0xFF == 27:
    cv.destroyAllWindows()
