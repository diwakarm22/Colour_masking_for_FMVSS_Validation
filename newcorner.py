import cv2
import numpy as np


def index_corners_with_delaunay(points):
    subdiv = cv2.Subdiv2D((0, 0, 100, 100))  # Provide an initial bounding box
    for pt in points:
        subdiv.insert((int(pt[0]), int(pt[1])))
    triangle_list = subdiv.getTriangleList()
    indexed_corners = []
    for i, pt in enumerate(points):
        for j, triangle in enumerate(triangle_list):
            if pt[0] == triangle[0] and pt[1] == triangle[1]:
                indexed_corners.append((i, j))
    return indexed_corners


def compute_homography(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=81, qualityLevel=0.01, minDistance=10)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                               criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    if corners is not None:
        img_with_corners = img.copy()
        for i, corner in enumerate(corners):
            x, y = corner.ravel()
            cv2.circle(img_with_corners, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.putText(img_with_corners, str(i), (int(x) - 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

        indexed_corners = index_corners_with_delaunay(corners.squeeze())
        print("Indexed Corners:", indexed_corners)

        cv2.imshow("Corners Detected", img_with_corners)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        real_world_coords = np.array([[-1.5, -1.5], [1.5, -1.5], [-1.5, 1.5], [1.5, 1.5]], dtype=np.float32)
        src_pts = np.array(corners.squeeze(), dtype=np.float32)

        homography_matrix, _ = cv2.findHomography(src_pts, real_world_coords, method=cv2.RANSAC)

        return homography_matrix
    else:
        return None


# Example usage
image_path = "C:/Users/uig40328/Downloads/attachments"
homography_matrix = compute_homography(image_path)
