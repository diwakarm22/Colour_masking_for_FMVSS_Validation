import numpy as np
import cv2 as cv

# Load the image
img = cv.imread("C:/Users/uig40328/Downloads/attachments/1.jpg")

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Perform corner detection
corners = cv.goodFeaturesToTrack(gray, maxCorners=81, qualityLevel=0.1, minDistance=12)

# Refine corner locations for better accuracy
corners = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))

# Display the result
if corners is not None:
    img_with_corners = img.copy()
    corners = np.intp(corners)

    # Sort the corners by their x-coordinate, then y-coordinate
    sorted_corners = sorted(corners, key=lambda corner: (corner[0][0], corner[0][1]))

    # Select the four inner corner points
    top_left = sorted_corners[10]
    top_right = sorted_corners[-11]
    bottom_left = sorted_corners[16]
    bottom_right = sorted_corners[-17]

    # Store the (u, v) pixel positions of the selected corners
    selected_corner_pixels = []
    selected_corners = [top_left, top_right, bottom_right, bottom_left]
    for i, corner in enumerate(selected_corners):
        u, v = corner.ravel()
        selected_corner_pixels.append((u, v))

    # Iterate over the selected four corners and draw circles with numbers
    for i, corner in enumerate(selected_corners):
        x, y = corner.ravel()
        cv.circle(img_with_corners, (x, y), 2, (0, 0, 255), -1)
        cv.putText(img_with_corners, str(i+1), (x+5, y+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

    print("Selected corner pixel positions:")

    # for calculating the homography matrix " making

    for i, corner_pixels in enumerate(selected_corner_pixels):
        print(f"{i+1}. ({corner_pixels[0]}, {corner_pixels[1]})")

    real_world_coords = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

    # Source points (pixel coordinates of the selected corners)
    src_pts = np.array(selected_corner_pixels, dtype=np.float32)

    # Destination points (real-world coordinates)
    dst_pts = real_world_coords

    # Compute the homography matrix
    homography_matrix, _ = cv.findHomography(src_pts, dst_pts,method=cv.RANSAC)

    print("Homography Matrix:")
    print(homography_matrix)

    cv.imshow('Corners Detected', img_with_corners)
    cv.waitKey(0)
    cv.destroyAllWindows()