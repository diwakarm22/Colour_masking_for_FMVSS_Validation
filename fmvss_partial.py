import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def mask_white_color(img):
    if img is None:
        print("Error: no image found.")
        exit()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 171])
    upper_white = np.array([180, 33, 255])
    mask = cv2.inRange(img_hsv, lower_white, upper_white)
    kernel = np.ones((1, 1), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    output_img = cv2.bitwise_and(img, img, mask=mask_cleaned)
    edges = cv2.Canny(output_img, 60, 150)
    dilation_kernel = np.ones((4, 4), np.uint8)
    dilated_edges = cv2.dilate(edges, dilation_kernel, iterations=1)
    eroded_edges = cv2.erode(dilated_edges, dilation_kernel, iterations=1)
    lines = cv2.HoughLinesP(eroded_edges, 1, np.pi / 180, threshold=52, minLineLength=70, maxLineGap=37)
    mask_white_final = np.zeros_like(mask_cleaned)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if (36 <= angle <= 38) or (-41 <= angle <= -40):
                cv2.line(mask_white_final, (x1, y1), (x2, y2), 255, 2)

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
    """Validates if the blue mask is touching the white mask."""
    intersection = cv2.bitwise_and(mask_white, mask_blue)
    if np.any(intersection > 0):
        print("Fail.")
    else:
        print(" Pass")
def create_colored_mask(mask_white, mask_blue, img_shape):
    """Creates an image with the white and blue masks colored."""
    colored_mask = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    colored_mask[mask_white > 0] = [255, 255, 255]  # White
    colored_mask[mask_blue > 0] = [255, 0, 0]  # Blue
    return colored_mask

img = cv2.imread(r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\AU416_Q6_Auto 1\AU416_Q6_Auto\400_dual_target_57.04_0_-90_1016.11_0_0_0_0_0_0_0_0_0_0_-2.856_9.9.png")
mask_white = mask_white_color(img)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_blue = mask_blue_color(img_hsv)
validate_blue_touching_white(mask_white, mask_blue)
colored_mask = create_colored_mask(mask_white, mask_blue, img.shape)
plt.imshow(cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB))
plt.title('Colored Mask (White and Blue)')
plt.show()
