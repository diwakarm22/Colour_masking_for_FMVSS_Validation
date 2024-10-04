import numpy as np
import cv2 as cv

img = cv.imread("C:/Users/uig40328/Downloads/attachments/front1.JPG")

cv.imshow('img', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find corners in the checkerboard
chessboard_size = (8, 8)  # Define the size of your checkerboard
found, corners = cv.findChessboardCorners(gray, chessboard_size, None)

if found:
    # Draw the detected corners on the image
    cv.drawChessboardCorners(img, chessboard_size, corners, found)

    # Display the image with detected corners
    cv.imshow('Corners Detected', img)

    # Number of detected corners
    num_corners = len(corners)
    print("Number of corners detected:", num_corners)

    # Apply the corner highlighting
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 1, 0.01)
    dst = cv.dilate(dst, None)
    img[dst > 0.12 * dst.max()] = [0, 0, 255]
    cv.imshow('Highlighted Corners', img)

else:
    print("Checkerboard corners not found.")

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
