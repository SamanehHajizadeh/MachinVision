from cv2 import cv2

from readImages.rescale import rescale_Frame

# Read photo
# img = cv2.imread('pic/cat.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('pic/contour.png', cv2.IMREAD_COLOR)

# Write Photo
out = cv2.imwrite('output/sam.png', img)

if out:
    print('Image is successfully saved as file.')

cv2.imshow('cat', img)
# cv2.imshow('cat', img)

# Resize image
resize_image = rescale_Frame(img)
cv2.imshow('sam', resize_image)

# Convert to grey
gift = cv2.imread('pic/gift.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(gift, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

# Blur
cute = cv2.imread('pic/contour.png', cv2.IMREAD_COLOR)
blur = cv2.GaussianBlur(cute, (7, 7), cv2.BORDER_DEFAULT)
cv2.imshow('Blur', blur)

# Edge Cascade
canny = cv2.Canny(blur, 125, 175)
cv2.imshow('Canny Edges', canny)

# Dilating the image
dilated = cv2.dilate(canny, (7, 7), iterations=3)
cv2.imshow('Dilated', dilated)

# Eroding
eroded = cv2.erode(dilated, (7, 7), iterations=3)
cv2.imshow('Eroded', eroded)

# Resize
resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
cv2.imshow('Resized', resized)

# Cropping
cropped = img[50:200, 200:400]
cv2.imshow('Cropped', cropped)

# Detects contours in images
imgray_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray_gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.imshow('Contours', img)

# Read images
img1 = cv2.imread('pic/input1.png')
img2 = cv2.imread('pic/input2.png')


# image inputs with applied parameters
#dest_and = cv2.bitwise_and(img2, img1, mask=None)
#cv2.imshow('Bitwise And', dest_and)
cv2.imshow('Bitwise And', img2)


cv2.waitKey(0)
