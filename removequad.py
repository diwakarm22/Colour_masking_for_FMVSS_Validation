import os
import cv2
import numpy as np

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
    _,contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_img = np.zeros_like(img)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 120:
            continue
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            cv2.drawContours(dilated_mask, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)
    result_img = cv2.merge([dilated_mask, dilated_mask, dilated_mask])
    result_img = np.where(result_img == 255, 255, 0).astype(np.uint8)
    return result_img

def process_images_in_folder(folder_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error: Unable to load image {filename}.")
                continue
            processed_img = mask_white_color(img)

            if processed_img is not None:
                output_processed_img_path = os.path.join(output_folder_path, f'processed_{filename}')
                cv2.imwrite(output_processed_img_path, processed_img)

                print(f"Processed and saved image with white mask on black background for {filename}.")
input_folder_path = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\AU416_Q6_Auto 1\AU416_Q6_Auto"
output_folder_path = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\output_processed_images"
process_images_in_folder(input_folder_path, output_folder_path)
