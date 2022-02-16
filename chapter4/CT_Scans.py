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
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    mask = ndimage.gaussian_filter(mask, sigma=l / n_pts)
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    res = (mask > mask.mean()) & mask_outer
    # plt.imshow(res, cmap='gray')
    # plt.show()
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


def build_projection_operator(l_x, n_dir):
    """Compute the tomography design matrix.

        Parameters
        ----------

        l_x : int
            linear size of image array

        n_dir : int
            number of angles at which projections are acquired.

        Returns
        -------
        p : sparse matrix of shape (n_dir l_x, l_x**2)
        """
    X, Y = _generate_center_coordinates(l_x)
    angles = np.linspace(0, np.pi, n_dir, endpoint=False)
    data_inds, weights, camera_inds = [], [], []
    data_unravel_indices = np.arange(l_x ** 2)
    data_unravel_indices = np.hstack((data_unravel_indices,
                                      data_unravel_indices))
    for i, angle in enumerate(angles):
        Xrot = np.cos(angle) * X - np.sin(angle) * Y
        inds, w = _weights(Xrot, dx=1, orig=X.min())
        mask = (inds >= 0) & (inds < l_x)
        weights += list(w[mask])
        camera_inds_prev = inds[mask] + i * l_x
        camera_inds += list(camera_inds_prev.astype('int'))
        data_inds += list(data_unravel_indices[mask])

    proj_operator = sparse.coo_matrix((weights, (camera_inds, data_inds)))

    return proj_operator


if __name__ == "__main__":
    l = 128
    data = generate_synthetic_data()
    # plt.figure(figsize=(5, 5))
    # plt.imshow(data, cmap='gray')
    # plt.axis('off')
    # plt.savefig('./data.png')
    # plt.show()

    proj_operator = build_projection_operator(l, l // 7)
    # print(proj_operator.shape)
    proj_t = np.reshape(proj_operator.todense().A, (l // 7, l, l, l))
    # print(proj_t.shape)
    # plt.imshow(proj_t[3, 0], cmap='gray')
    # plt.show()
    # plt.imshow(proj_t[3, 1], cmap='gray')
    # plt.show()
    # plt.imshow(proj_t[3, 2], cmap='gray')
    # plt.show()
    # plt.imshow(proj_t[3, 40], cmap='gray')
    # plt.show()
    # plt.imshow(proj_t[4, 40], cmap='gray')
    # plt.show()
    # plt.imshow(proj_t[15, 40], cmap='gray')
    # plt.show()
    # plt.imshow(proj_t[17, 40], cmap='gray')
    # plt.show()

    proj = proj_operator @ data.ravel()[:, np.newaxis]
    # print(np.resize(proj, (l // 7, l))[3, 14])


    # plt.figure(figsize=(5, 5))
    # plt.imshow(data + proj_t[17, 40], cmap=plt.cm.gray)
    # plt.axis('off')
    # plt.show()
    # # plt.savefig("images/data_xray.png")
    # both = data + proj_t[17, 40]
    # plt.imshow((both > 1.1).astype(int), cmap=plt.cm.gray)
    # plt.show()
    proj += 0.15 * np.random.randn(*proj.shape)
    plt.figure(figsize=(7, 7))
    plt.imshow(np.resize(proj, (l // 7, l)), cmap='gray')
    plt.axis('off')
    plt.show()

    print(proj.shape)