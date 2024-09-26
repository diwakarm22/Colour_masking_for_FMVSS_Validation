import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mask_white_color(img):
    """Creates a mask for white color in the image."""
    if img is None:
        print("Error: no image found.")
        exit()
    # Convert the image to HSV color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Refine the range for white color in HSV
    lower_white = np.array([0, 0, 171])  # Increased to focus on more pure white
    upper_white = np.array([180, 33, 255])  # Decreased saturation upper bound to exclude light blues
    # Create a mask
    mask = cv2.inRange(img_hsv, lower_white, upper_white)
    # Apply morphological operations to clean up the mask
    kernel = np.ones((1, 1), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Apply the cleaned mask to the original image
    output_img = cv2.bitwise_and(img, img, mask=mask_cleaned)
    # Perform edge detection
    edges = cv2.Canny(output_img, 60, 150)
    # Apply dilation followed by erosion (closing operation)
    dilation_kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, dilation_kernel, iterations=1)
    eroded_edges = cv2.erode(dilated_edges, dilation_kernel, iterations=1)
    lines = cv2.HoughLinesP(eroded_edges, 1, np.pi / 180, threshold=65, minLineLength=50, maxLineGap=40)
    # Create the final white mask
    mask_white_final = np.zeros_like(mask_cleaned)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Check if the angle is within the desired ranges
            if (36 <= angle <= 38) or (-41 <= angle <= -40):
                cv2.line(mask_white_final, (x1, y1), (x2, y2), 255, 1)  # Draw on the mask

    return mask_white_final

def mask_blue_color(img_hsv):
    """Creates a mask for blue color in the image."""
    lower_blue1 = np.array([82, 105, 105])
    upper_blue1 = np.array([120, 255, 255])
    lower_blue2 = np.array([84, 60, 60])
    upper_blue2 = np.array([120, 180, 255])
    mask_blue1 = cv2.inRange(img_hsv, lower_blue1, upper_blue1)
    mask_blue2 = cv2.inRange(img_hsv, lower_blue2, upper_blue2)
    return cv2.bitwise_or(mask_blue1, mask_blue2)

def validate_blue_touching_white(mask_white, mask_blue):
    """Validates if the blue mask is touching the white mask and counts the number of touching pixels."""
    # Perform bitwise AND between white and blue masks
    intersection = cv2.bitwise_and(mask_white, mask_blue)
    # Count the number of touching pixels
    touching_pixels = np.count_nonzero(intersection)
    # Check if there are any non-zero pixels in the intersection
    if touching_pixels > 0:
        print(f"Fail. {touching_pixels} pixels are touching.")
    else:
        print("Pass. No pixels are touching.")

def create_colored_mask(mask_white, mask_blue, img_shape):
    """Creates an image with the white and blue masks colored."""
    colored_mask = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    colored_mask[mask_white > 0] = [255, 255, 255]  # White
    colored_mask[mask_blue > 0] = [255, 0, 0]  # Blue
    return colored_mask

# Load the image
img = cv2.imread(r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\failed.png")

# Create masks for white and blue colors
mask_white = mask_white_color(img)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_blue = mask_blue_color(img_hsv)

# Validate if the blue mask is touching the white mask and count the number of touching pixels
validate_blue_touching_white(mask_white, mask_blue)

# Create and display the colored mask
colored_mask = create_colored_mask(mask_white, mask_blue, img.shape)
plt.imshow(cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB))
plt.title('Colored Mask (White and Blue)')
plt.show()
