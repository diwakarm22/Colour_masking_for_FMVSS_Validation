import cv2
import numpy as np


def mask_white_color(img_hsv):
    """Creates a mask for white color in the image, including two color ranges."""
    # First range for pure white
    lower_white1 = np.array([0, 0, 171])
    upper_white1 = np.array([180, 33, 255])

    # Second range for near-white (e.g., light gray or sky blue mixed with white)
    lower_white2 = np.array([85, 40, 160])
    upper_white2 = np.array([115, 170, 255])

    # Create masks for both ranges
    mask_white1 = cv2.inRange(img_hsv, lower_white1, upper_white1)
    mask_white2 = cv2.inRange(img_hsv, lower_white2, upper_white2)

    # Combine the two masks using bitwise OR
    mask_white = cv2.bitwise_or(mask_white1, mask_white2)

    # Apply morphological closing to clean up the mask
    kernel = np.ones((2, 2), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

    # Display the cleaned mask
    cv2.imshow("mask_cleaned", mask_cleaned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Load the image from a specific path
image_path = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\Capture.JPG"  # Replace with your image path
img = cv2.imread(image_path)

# Check if the image was loaded successfully
if img is None:
    print(f"Failed to load image from {image_path}")
else:
    # Convert the image to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Apply the mask function
    mask_white_color(img_hsv)


