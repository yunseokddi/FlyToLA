import numpy as np
import cv2

# a = np.array([1., 2., 3.])
# b = 2.
#
# print(a * b)
# print(a + b)
# print(a - b)
# print(a / b)

# a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# # print(a.shape)
#
# # axis 0
# a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
#
# print('axis 0')
# print(np.sum(a, axis=0))
# print('axis 1')
# print(np.sum(a, axis=1))
# print('axis 2')
# print(np.sum(a, axis=2))

img = cv2.imread('./cat.jpeg')
scale = np.array([2, 2, 2])

print(img.shape)
print(scale.shape)

new_image = img * scale
print(new_image.shape)
cv2.imwrite('cat_1.jpeg', new_image)
