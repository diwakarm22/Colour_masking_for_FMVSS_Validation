import os
import cv2
import numpy as np
import pandas as pd
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

    # Find contours in the mask
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
    lines = cv2.HoughLinesP(dilated_mask, 1, np.pi / 360, threshold=52, minLineLength=70, maxLineGap=37)

    # Create an empty image to draw lines
    lines_mask = np.zeros_like(dilated_mask)

    # Draw non-horizontal lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if not (-15 < angle < 15):  # Exclude horizontal lines with a tolerance of 10 degrees
                cv2.line(lines_mask, (x1, y1), (x2, y2), 255, 1)

    # Return both the mask and the lines
    return lines_mask

"""Creates a mask for blue color in the image."""
def mask_blue_color(img_hsv):
    lower_blue1 = np.array([90, 50, 50])
    upper_blue1 = np.array([130, 255, 255])
    lower_blue2 = np.array([82, 52, 129])
    upper_blue2 = np.array([102, 132, 209])
    mask_blue1 = cv2.inRange(img_hsv, lower_blue1, upper_blue1)
    mask_blue2 = cv2.inRange(img_hsv, lower_blue2, upper_blue2)
    return cv2.bitwise_or(mask_blue1, mask_blue2)
def validate_overlay_touching_whiteline(mask_white, mask_blue):
    intersection = cv2.bitwise_and(mask_white, mask_blue)
    touching_pixels = np.count_nonzero(intersection)
    if touching_pixels == 0:
        return "Pass", touching_pixels
    elif 1 <= touching_pixels <= 5:
        return "BPass", touching_pixels
    else:
        return "Fail", touching_pixels
def create_colored_mask(mask_white, mask_blue, img_shape):
    colored_mask = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    colored_mask[mask_white > 0] = [255, 255, 255]  # White
    colored_mask[mask_blue > 0] = [255, 0, 0]  # Blue
    return colored_mask
def process_images_in_folder(folder_path, csv_file_path, fail_save_folder):
    df = pd.read_csv(csv_file_path, encoding='ISO-8859-1') if os.path.exists(csv_file_path) else pd.DataFrame(
        columns=['Image Name', 'Result', 'pixels touching'])
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            mask_white = mask_white_color(img)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask_blue = mask_blue_color(img_hsv)
            result, touching_pixels = validate_overlay_touching_whiteline(mask_white, mask_blue)
            colored_mask = create_colored_mask(mask_white, mask_blue, img.shape)
            df = pd.concat([df, pd.DataFrame(
                {'Image Name': [filename], 'Result': [result], 'pixels touching': [touching_pixels]})],
                           ignore_index=True)
            #if you don't want to store failed images remove these two lines
            if "Fail" in result:
                cv2.imwrite(os.path.join(fail_save_folder, filename), colored_mask)


    df.to_csv(csv_file_path, index=False)

folder_path = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\AU416_Q6_Auto_FOV_till_1278"
csv_file_path = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\1278.csv"
fail_save_folder = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\failed_images"
process_images_in_folder(folder_path, csv_file_path, fail_save_folder)