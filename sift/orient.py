import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid


def sift_localize(img_truth, img_scene):
    """Localize `img_scene` to `img_truth` using SIFT."""

    # Initiate SIFT keypoint detector
    descriptor = cv2.SIFT_create()
    kp_truth, des_truth = descriptor.detectAndCompute(img_truth, None)
    kp_scene, des_scene = descriptor.detectAndCompute(img_scene, None)

    # FLANN parameters (trust me, otherwise this doesnt work according to doc for some reason)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    feature_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    matches = feature_matcher.knnMatch(des_scene, des_truth, k=2)
    nn_match_ratio = 0.7
    good_matches = [m for m, n in matches if m.distance < nn_match_ratio * n.distance]

    # homography using RANSAC
    if len(good_matches) < 10:
        raise ValueError(f"Length of good matches is {len(good_matches)}")

    truth_pts = np.float32([kp_truth[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    scene_pts = np.float32([kp_scene[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(scene_pts, truth_pts, cv2.RANSAC, 5.0)
    img_corrected = cv2.warpPerspective(img_scene, homography, img_truth.shape)

    return img_corrected


def find_circles(img):
    """Find circles in img"""

    blurred = cv2.medianBlur(img, ksize=5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,  # inverse ratio of resolution
        20,  # min distance between detected centers
        param1=200,  # upper threshold for internal canny edge detector
        param2=100,  # threshold for center detection
        minRadius=100,
        maxRadius=0
    )
    if circles is None:
        return []

    circles = circles[0, :]
    return circles


UNIT_SQUARE_RADIUS = 220
EMPTY_HOLE_PERCENT = 0.3


def find_unit_coords_rough(img):
    """Find rough centroids to units in the board.

    Returns: list of (x, y) centroids
    """
    circles = find_circles(img)
    circles_no_radius = circles[:, 0:2]

    cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=150, metric="euclidean").fit_predict(
        circles_no_radius)
    cluster_centroids = NearestCentroid().fit(circles_no_radius, cluster_labels).centroids_

    units = []
    for x, y in np.uint16(np.around(cluster_centroids)):
        cropped = img[y - UNIT_SQUARE_RADIUS:y + UNIT_SQUARE_RADIUS, x - UNIT_SQUARE_RADIUS:x + UNIT_SQUARE_RADIUS]
        _, cropped_binarized = cv2.threshold(cropped, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        black_to_white_ratio = np.sum(cropped_binarized) / (255 * 4 * UNIT_SQUARE_RADIUS ** 2)

        if black_to_white_ratio < EMPTY_HOLE_PERCENT:
            units.append((x, y))

    return np.array(units)


TRUTH_LOCALIZATION = cv2.imread("dataset/correct.png", cv2.IMREAD_GRAYSCALE)


def find_unit_imgs(img):
    """Get a list of cropped rotated images with the units in the board"""

    coords = find_unit_coords_rough(img)
    unit_imgs = []

    for x, y in coords:
        cropped = img[y - UNIT_SQUARE_RADIUS:y + UNIT_SQUARE_RADIUS, x - UNIT_SQUARE_RADIUS:x + UNIT_SQUARE_RADIUS]
        localized = cropped

        unit_imgs.append(localized)

    return unit_imgs


img_path = "dataset/cyrcle.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
cv2.imwrite("output/sam11.png", img)
plt.imshow(img)
plt.show()

unit_imgs = find_unit_imgs(img)

for unit in unit_imgs:
    #cv2.imwrite("output/sam1.png", unit)
    plt.imshow(unit, cmap="gray")
    plt.show()
