import numpy as np
import cv2 as cv

# Load the image
img = cv.imread("C:/Users/uig40328/Downloads/attachments/front1.jpg")

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Threshold the image to create a binary mask
_, binary = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assuming it's the checkerboard)
max_area = 0
max_contour = None
for contour in contours:
    area = cv.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

# Create a mask for the checkerboard region
mask = np.zeros_like(gray)
cv.drawContours(mask, [max_contour], -1, 255, cv.FILLED)

# Extract the checkerboard region (ROI)
roi = cv.bitwise_and(img, img, mask=mask)

# Create a green background image
green_bg = np.zeros_like(img)
green_bg[:] = (0, 255, 0)  # Set the color to green (BGR)

# Blend the ROI and green background
result = cv.bitwise_or(roi, green_bg, mask=cv.bitwise_not(mask))

# Perform corner detection within the checkerboard region
gray_roi = cv.bitwise_and(gray, gray, mask=mask)
corners = cv.goodFeaturesToTrack(gray_roi, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners = cv.cornerSubPix(gray_roi, corners, (5, 5), (-1, -1), criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))

# Display the result
if corners is not None:
    for corner in corners:
        x, y = corner.ravel()
        cv.circle(result, (x, y), 2, (0, 0, 255), -1)

    cv.imshow('Corners Detected', result)

    # Wait for user input to close the window
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("No corners detected.")