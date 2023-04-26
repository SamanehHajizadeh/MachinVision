import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

img_bgr = cv2.imread("dataset/correct.png")

img_truth = cv2.imread('img/truth.png', cv2.IMREAD_GRAYSCALE)  # truth image (searched pattern)
img_scene = cv2.imread('img/2.png', cv2.IMREAD_GRAYSCALE)  # query image (pattern in a scene)

start_time = time.time()

sift = cv2.SIFT_create()

kp_truth, des_truth = sift.detectAndCompute(img_truth, None)
kp_scene, des_scene = sift.detectAndCompute(img_scene, None)

"""
algorithm takes an integer, what is missing from the tutorial is the initialisation of FLANN_INDEX_KDTREE and FLANN_INDEX_LSH with different values. (The upper case should have been a hint that these are meant as descriptive labels of fixed integer values.
"""
FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

"""
FlannBasedMatcher() calls its nearest search methods to find the best matches
"""
feature_matcher = cv2.FlannBasedMatcher(index_params, search_params)

"""
 Finds the k best matches for each descriptor from a query set.
"""
matches = feature_matcher.knnMatch(des_scene, des_truth, k=2)

nn_match_ratio = 0.7
good_matches = [m for m, n in matches if m.distance < nn_match_ratio * n.distance]

img_matches = cv2.drawMatchesKnn(img_truth,
                                 kp_truth,
                                 img_scene,
                                 kp_scene,
                                 [[m] for m in good_matches],
                                 None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img_matches, cmap="gray")
plt.show()

# homography using RANSAC
if len(good_matches) < 10:
    raise ValueError(f"Length of good matches is {len(good_matches)}")

truth_pts = np.float32([kp_truth[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
scene_pts = np.float32([kp_scene[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

homography, _ = cv2.findHomography(scene_pts, truth_pts, cv2.RANSAC, 5.0)
img_corrected = cv2.warpPerspective(img_scene, homography, img_truth.shape)
plt.imshow(img_corrected, cmap="gray")
plt.show()

end_time = time.time()

print(end_time - start_time, "s")

img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

HUE = 21
SATURATION = 148
VALUE = 166
AREA_THRESHOLD = 10

yellow_mask = cv2.inRange(
    img_hsv,
    (HUE - 5, SATURATION - 45, VALUE - 45),
    (HUE + 5, SATURATION + 45, VALUE + 45)
)

erosion_size = 1
structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                                (erosion_size, erosion_size))

yellow_mask = cv2.erode(yellow_mask, structuring_element, iterations=4)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(yellow_mask, None, None, None, 8, cv2.CV_32S)
component_areas = stats[1:, cv2.CC_STAT_AREA]
for i in range(0, nlabels - 1):
    x, y = centroids[i]
    if component_areas[i] >= AREA_THRESHOLD:
        if x > yellow_mask.shape[0] / 2:
            x_quad = "right"
        else:
            x_quad = "left"

        if y > yellow_mask.shape[1] / 2:
            y_quad = "top"
        else:
            y_quad = "bottom"

        print("Missing solder ", x_quad, y_quad)

plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(yellow_mask, cmap="gray")
plt.show()
