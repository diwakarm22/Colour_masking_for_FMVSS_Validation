import numpy as np
import cv2 as cv

# Load the image
img = cv.imread("C:/Users/uig40328/Downloads/attachments/Capture6.JPG")

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Define the checkerboard size
max_rows = 8
max_cols = 8

# Try detecting the checkerboard corners
ret, corners = cv.findChessboardCorners(gray, (max_cols, max_rows), None)

# If corners are found, refine them and mark them
if ret:
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Draw the refined corners
    for corner in corners:
        x, y = corner.ravel()  # Extract x and y coordinates
        cv.circle(img, (x, y), 5, (0, 0, 255), -1)  # Mark corner with a circle

    # Display the result with corners marked for all squares
    cv.imshow('Corners Marked for All Squares', img)

# Wait for user input to close the window
cv.waitKey(0)
cv.destroyAllWindows()
