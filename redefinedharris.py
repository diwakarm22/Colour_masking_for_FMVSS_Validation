import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


img = cv.imread("C:/Users/uig40328/Downloads/blenderout/img1.png")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges =cv.Canny(img,100,200)
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    # Approximate the contour to a polygon
    approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)

    # Check if the contour has 4 vertices (a quadrilateral)
    if len(approx) == 4:
        # Assume this is the chessboard contour
        x, y, w, h = cv.boundingRect(approx)
        roi = img[y:y+h, x:x+w]
        break
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(roi, cmap='gray')
plt.title('Extracted Chessboard ROI')
plt.axis('off')

plt.show()