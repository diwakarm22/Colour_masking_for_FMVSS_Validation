import cv2
import numpy as np
def detect_and_remove_quadrilateral(input_image_path, output_image_path):
    img = cv2.imread(input_image_path)
    if img is None:
        print("Error: no image found.")
        exit()

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 171])  # Increased to focus on more pure white
    upper_white = np.array([180, 33, 255])  # Decreased saturation upper bound to exclude light blues
    # Create a mask
    mask = cv2.inRange(img_hsv, lower_white, upper_white)
    kernel = np.ones((1, 1), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    output_img = cv2.bitwise_and(img, img, mask=mask_cleaned)
    edges = cv2.Canny(output_img, 60, 150)
    edge_detected_image = edges
    dilation_kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, dilation_kernel, iterations=1)
    eroded_edges = cv2.erode(dilated_edges, dilation_kernel, iterations=1)
    lines = cv2.HoughLinesP(eroded_edges, 1, np.pi / 180, threshold=65, minLineLength=50, maxLineGap=40)
    mask_white_final = np.zeros_like(mask_cleaned)
    if edge_detected_image is None:
        print("Error: The image could not be loaded.")
        exit()

    _,contours, _ = cv2.findContours(edge_detected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the quadrilateral
    mask = np.zeros_like(edge_detected_image)
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        if area < 1000:
            continue
        if len(approx) == 4:
            cv2.drawContours(mask, [contour], -1, 255, -1)  # Fill the contour with white

    mask_inv = cv2.bitwise_not(mask)

    # Apply the mask to the edge-detected image to remove the quadrilateral
    new_edge_detected_image = cv2.bitwise_and(edge_detected_image, mask_inv)
    dilation_kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(new_edge_detected_image, dilation_kernel, iterations=1)
    eroded_edges = cv2.erode(dilated_edges, dilation_kernel, iterations=1)

    lines = cv2.HoughLinesP(eroded_edges, 1, np.pi / 180, threshold=65, minLineLength=50, maxLineGap=40)
    img_with_lines = np.copy(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Save and display the final images using OpenCV
    # Save and display the final images using OpenCV
    cv2.imwrite(output_image_path, new_edge_detected_image)

    # Display images using OpenCV
    cv2.imshow('Final Edge-Detected Image', eroded_edges)
    cv2.imshow('Hough Lines', img_with_lines)

    # Save the final edge-detected image with a proper extension
    cv2.imwrite('Final_Edge_Detected_Image.png', new_edge_detected_image)

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


input_image_path = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\input\failed.png"
output_image_path = r"D:\Conti_Blender\Conti_Blender\Projects\Working_scripts\output\result.png"

detect_and_remove_quadrilateral(input_image_path, output_image_path)
