import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

img_bgr = cv2.imread("dataset/correct.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
_, img_binary = cv2.threshold(img_gray, 75, 255, cv2.THRESH_BINARY)
img_edges = cv2.Canny(img_gray, 100, 250)

MIN_COMPONENT_AREA = 4000
MIN_CONTOUR_AREA = 1000

# binary image filtering
erosion_size = 1
structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                                (erosion_size, erosion_size))
img_binary_eroded = cv2.erode(img_binary, structuring_element, iterations=13)
img_binary_eroded = cv2.dilate(img_binary_eroded, structuring_element, iterations=15)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_binary_eroded, None, None, None, 8, cv2.CV_32S)
component_areas = stats[1:, cv2.CC_STAT_AREA]
img_binary_filtered = np.zeros((labels.shape), np.uint8)
for i in range(0, nlabels - 1):
    if component_areas[i] >= MIN_COMPONENT_AREA:
        img_binary_filtered[labels == i + 1] = 255

# show binary images
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
ax0.imshow(img_binary, cmap="gray")
ax0.set_title("img_binary")
ax1.imshow(img_binary_eroded, cmap="gray")
ax1.set_title("img_binary_eroded")
ax2.imshow(img_binary_filtered, cmap="gray")
ax2.set_title("img_binary_filtered")
plt.show()

# ax2.imshow(img_edges, cmap="gray")
# plt.show()

# circles = cv2.HoughCircles(
#     img_edges,
#     cv2.HOUGH_GRADIENT,
#     1,           # inverse ratio of resolution
#     20,          # min distance between detected centers
#     param1=1,    # upper threshold for internal canny edge detector
#     param2=75,   # threshold for center detection
#     minRadius=200,
#     maxRadius=400
# )

lines = cv2.HoughLines(
    img_edges,
    1,  # rho resolution
    math.pi / 180,  # theta resolution
    131,  # threshold
)

contours, hierarchy = cv2.findContours(
    img_binary_filtered,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

# lines = cv2.HoughLinesP(
#     img_edges,
#     1,           # rho resolution
#     math.pi/180, # theta resolution
#     50,          # threshold
#     100,        # min number of points to detect a line
#     10           # max gap between points to be considered a line
# )

img_contoured = img_rgb.copy()

if contours is not None:
    for i, contour in enumerate(contours):
        cv2.drawContours(img_contoured, [contour], -1, (255 - i, 0, i), 3)
    # cv2.drawContours(img_contoured, contours, -1, (0, 0, 255), 3)

    rectangles = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            rectangles.append(approx)
    # cv2.drawContours(img_contoured, rectangles, -1, (0,255,255), 5)

if lines is not None:
    for line in lines:
        for rho, theta in line:
            a = math.cos(theta)
            b = math.sin(theta)
            xc = a * rho
            yc = b * rho
            x1 = int(xc + 1000 * (-b));
            y1 = int(yc + 1000 * (a));
            x2 = int(xc - 1000 * (-b));
            y2 = int(yc - 1000 * (a));
            cv2.line(img_contoured, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # for x1, y1, x2, y2 in line:
        #     cv2.line(img_contoured, (x1, y1), (x2, y2), (255, 0, 0), 2)

plt.imshow(img_contoured)
plt.show()
end_time = time.time()
