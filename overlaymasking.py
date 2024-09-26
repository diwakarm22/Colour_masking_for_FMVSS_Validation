import cv2
import numpy as np

img = cv2.imread(r'D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\in2.png')
if img is None:
    print("Error: Unable to load image.")
    exit()

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([88, 43, 43])
upper_blue = np.array([138, 255, 255])
lower_blue1 = np.array([82, 52, 129])
upper_blue1 = np.array([102, 132, 209])
mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
mask1 = cv2.inRange(img_hsv, lower_blue1, upper_blue1)
mask_combined = cv2.bitwise_or(mask, mask1)
kernel = np.ones((1, 1), np.uint8)
mask_closed = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
output_img = cv2.bitwise_and(img, img, mask=mask_closed)
output_path = r'D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\AU416_Q6_Auto 1\AU416_Q6_Auto\400_dual_target_57.04_0_-90_1016.11_2_2_0_5_0_0_0_0_0_0_8.105_-4.963.png'
cv2.imwrite(output_path, output_img)
cv2.imshow("Masked Image", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Image saved at {output_path}")
