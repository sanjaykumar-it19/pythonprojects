import cv2
image = cv2.imread("testimg.jpg")
print(image.size)
print(image.shape)
print(image.dtype)
updatedimg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imwrite("grayimage.jpg",updatedimg)
cv2.imshow("gray image",updatedimg)
cv2.imshow("original image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
