import cv2
import numpy as np
import matplotlib.pyplot as plt

def mask_white_color(img):
    if img is None:
        print("Error: no image found.")
        exit()

    # Convert image to HSV and create a white mask
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 171])
    upper_white = np.array([180, 33, 255])
    mask = cv2.inRange(img_hsv, lower_white, upper_white)
    kernel = np.ones((1, 1), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Dilation and erosion on the binary mask
    dilation_kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask_cleaned, dilation_kernel, iterations=1)
    #eroded_mask = cv2.erode(dilated_mask, dilation_kernel, iterations=1)

    # Line detection on the eroded mask
    lines = cv2.HoughLinesP(dilated_mask, 1, np.pi / 180, threshold=75, minLineLength=70, maxLineGap=40)

    # Prepare a final mask to draw the filtered lines
    mask_white_final = np.zeros_like(mask_cleaned)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Check if the angle is within the desired ranges
            if (37 <= angle <= 38) or (-42 <= angle <= -39):
                cv2.line(mask_white_final, (x1, y1), (x2, y2), 255, 2)

    # Display the mask with the detected lines
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_white_final, cmap='gray')
    plt.title("Mask with Filtered Lines")
    plt.axis('off')
    plt.show()

    return mask_white_final

# Load the image
img_path = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\in2.png"
img = cv2.imread(img_path)

# Call the function and visualize the output image
output_img = mask_white_color(img)
