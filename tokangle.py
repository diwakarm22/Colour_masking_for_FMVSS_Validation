import cv2
import numpy as np
def random_color():
    """Generate a random color."""
    return list(np.random.randint(0, 255, 3))
image = cv2.imread('path_to_your_image')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        color = random_color()
        cv2.line(image, (x1, y1), (x2, y2), color, 3)
        print(f"Line between ({x1},{y1}) and ({x2},{y2}) has an angle of {angle:.2f} degrees. Color: {color}")

else:
    print("No lines detected")

cv2.imshow('Detected Lines with Unique Colors', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
