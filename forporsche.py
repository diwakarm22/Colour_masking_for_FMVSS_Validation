import os
import cv2
import numpy as np
import pandas as pd
def mask_white_color(img):
    if img is None:
        print("Error: no image found.")
        return None
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 171])
    upper_white = np.array([180, 33, 255])
    mask = cv2.inRange(img_hsv, lower_white, upper_white)
    kernel = np.ones((1, 1), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    dilation_kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask_cleaned, dilation_kernel, iterations=1)
    _, contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            cv2.drawContours(dilated_mask, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)
    lines = cv2.HoughLinesP(dilated_mask, 1, np.pi /360, threshold=52, minLineLength=70, maxLineGap=37)
    lines_mask = np.zeros_like(dilated_mask)  # Create a black image of the same size
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_mask, (x1, y1), (x2, y2), 255, 1)  # Draw white lines on a black background
    return lines_mask

"""Creates a mask for blue color in the image."""
def mask_blue_color(img_hsv):
    lower_blue1 = np.array([82, 105, 105])
    upper_blue1 = np.array([120, 255, 255])
    lower_blue2 = np.array([84, 60, 60])
    upper_blue2 = np.array([120, 180, 255])
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

folder_path = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\inputimages"
csv_file_path = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\results1.csv"
fail_save_folder = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\failed_images"
process_images_in_folder(folder_path, csv_file_path, fail_save_folder)