import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import ndimage, sparse


# Generate Data
def generate_synthetic_data():
    rs = np.random.RandomState(0)
    n_pts = 36
    x, y = np.ogrid[0:l, 0:l]  # x.shape: (128,1) y.shape: (1,128)
    mask_outer = (x - l / 2) ** 2 + (y - l / 2) ** 2 < (l / 2) ** 2  # If elements are less than 64^2, set the False
    mx, my = rs.randint(0, l, (2, n_pts))  # mx.shape: (36,) my.shape: (36,)
    mask = np.zeros((l, l))
    mask[mx, my] = 1
    plt.imshow(mask, cmap='gray')
    plt.show()
    mask = ndimage.gaussian_filter(mask, sigma=l / n_pts)
    plt.imshow(mask, cmap='gray')
    plt.show()
    res = (mask > mask.mean()) & mask_outer
    plt.imshow(res, cmap='gray')
    plt.show()
    return res ^ ndimage.binary_erosion(res)


# Generate Projections
def _weights(x, dx=1, orig=0):
    x = np.ravel(x)
    floor_x = np.floor((x - orig) / dx)
    alpha = (x - orig - floor_x * dx) / dx
    return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))


def _generate_center_coordinates(l_x):
    X, Y = np.mgrid[:l_x, :l_x].astype(np.float64)
    center = l_x / 2.
    X += 0.5 - center
    Y += 0.5 - center
    return X, Y


if __name__ == "__main__":
    l = 128
    data = generate_synthetic_data()
    plt.figure(figsize=(5, 5))
    plt.imshow(data, cmap='gray')
    plt.show()
