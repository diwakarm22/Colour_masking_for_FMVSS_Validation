import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread(r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\failed.png")
if img is None:
    print("Error: no image found.")
    exit()
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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
edges = cv2.Canny(output_img, 50, 150)
# Display the edge-detected image
plt.imshow(edges, cmap='gray')
plt.title('Edge Detected Image')
plt.show()
# Perform Hough line detection
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=50)
# Copy the original image to draw the lines on it
img_with_lines = np.copy(img)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        # Check if the angle is within the desired ranges
        if (36 <= angle <= 38) or (-38 <= angle <= -36):
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)
            print(f"Line from ({x1}, {y1}) to ({x2}, {y2}) with angle {angle:.2f} degrees kept.")
# Display the filtered image with lines
plt.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
plt.title('Filtered Hough Lines')
plt.show()
# Save the output image with filtered Hough lines to a specific path
output_path = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\output\image_with_filtered_lines.png"
output_path1 = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\output\edges.png"
cv2.imwrite(output_path, img_with_lines)
cv2.imwrite(output_path1, edges)
