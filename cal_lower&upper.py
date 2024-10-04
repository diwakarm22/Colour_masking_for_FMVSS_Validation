import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread(r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\image.png")
if img is None:
    print("Error: no image found.")
    exit()

# Convert the image to HSV color space
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the range of colors you want to keep (e.g., white color)
lower_white = np.array([0, 0, 171])  # Lower bound for white
upper_white = np.array([180, 33, 255])  # Upper bound for white

# Create a binary mask where white represents the colors in the range
binary_mask = cv2.inRange(img_hsv, lower_white, upper_white)

# Optionally, you can apply a threshold to the HSV image directly
# ret, binary_img = cv2.threshold(img_hsv[:,:,2], 127, 255, cv2.THRESH_BINARY)

# Display the binary mask
plt.imshow(binary_mask, cmap='gray')
plt.title('Binary Image')
plt.show()

# Save the binary image
output_path = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\output\binary_image.png"
#cv2.imwrite(output_path, binary_mask)
