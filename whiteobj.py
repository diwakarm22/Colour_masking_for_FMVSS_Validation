import os
import cv2
import numpy as np

def mask_white_color(img):
    if img is None:
        print("Error: no image found.")
        return None

    # Convert image to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define white color range in HSV
    lower_white = np.array([0, 0, 171])
    upper_white = np.array([180, 33, 255])

    # Create mask to isolate white colors
    mask = cv2.inRange(img_hsv, lower_white, upper_white)

    # Clean up the mask using morphology operations
    kernel = np.ones((1, 1), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Dilate the mask to fill gaps in the contours
    dilation_kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask_cleaned, dilation_kernel, iterations=1)

    # Edge detection using Canny
    edges = cv2.Canny(dilated_mask, 100, 200)

    # Find contours in the edges
    _,contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each contour and filter based on area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 120:  # Adjust this threshold as needed
            continue

        # Approximate contour to reduce the number of points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the contour is a quadrilateral (4 points), remove it
        if len(approx) == 4:
            cv2.drawContours(dilated_mask, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)

    # Use HoughLinesP to detect lines on the remaining mask
    lines = cv2.HoughLinesP(dilated_mask, 1, np.pi / 360, threshold=72, minLineLength=70, maxLineGap=37)

    # Create an empty image to draw lines
    lines_mask = np.zeros_like(dilated_mask)

    # Draw non-horizontal lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if not (-15 < angle < 15):  # Exclude horizontal lines with a tolerance of 15 degrees
                cv2.line(lines_mask, (x1, y1), (x2, y2), 255, 1)

    # Return the lines mask
    return lines_mask


def process_images_in_folder(folder_path, output_folder_path):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Loop through all images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error: Unable to load image {filename}.")
                continue

            # Apply the white mask function
            lines_mask = mask_white_color(img)

            if lines_mask is not None:
                # Save the lines_mask to the output folder
                output_lines_mask_path = os.path.join(output_folder_path, f'lines_mask_{filename}')
                cv2.imwrite(output_lines_mask_path, lines_mask)  # Add the missing argument

                print(f"Processed and saved lines mask for {filename}.")


# Folder paths
input_folder_path = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\AU416_Q6_Auto 1\AU416_Q6_Auto"
output_folder_path = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\output_lines_masks"

# Process all images in the folder and save their lines masks
process_images_in_folder(input_folder_path, output_folder_path)
