import cv2

image = cv2.imread('test.jpg')

print(f"width: {image.shape[1]} pixels")
print(f"height: {image.shape[0]} pixels")
print(f"channels: {image.shape[2]}")

cv2.imshow("Image",image)
cv2.waitKey(0)

cv2.imwrite("new_image.jpg",image)