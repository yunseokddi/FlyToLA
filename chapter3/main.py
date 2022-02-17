import moviepy.editor as mpe
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from HelperMethods import *

TOL = 1.0e-8

video = mpe.VideoFileClip('./data/Video_003.avi')

# print(video.duration)

if __name__ == "__main__":
    scale = 25
    dims = (int(240 * (scale / 100)), int(320 * (scale / 100)))

    M = create_data_matrix_from_video(video, 100, scale)
    # print(dims, M.shape)
    # (60, 80), (4800, 11300)
    # plt.imshow(np.reshape(M[:, 140], dims), cmap='gray')
    # plt.show()
    # np.save("low_res_surveillance_matrix.npy", M)

    M = np.load('./data/low_res_surveillance_matrix.npy')
    # plt.figure(figsize=(12, 12))
    # plt.imshow(M, cmap='gray')
    # plt.show()

    # A first attempt with SVD

    # u, s, v = decomposition.randomized_svd(M, 2)

    # print(M.shape)  # (4800, 11300)
    # print(u.shape, s.shape, v.shape)  # (4800, 2) (2,) (2, 11300)

    # low_rank = u @ np.diag(s) @ v

    # print(low_rank.shape) # (4800, 11300)

    # plt.figure(figsize=(12, 12))
    # plt.imshow(low_rank, cmap='gray')
    # plt.show()

    # plt.imshow(np.reshape(low_rank[:, 140], dims), cmap='gray')
    # plt.show()
    #
    # plt.imshow(np.reshape(M[:, 550] - low_rank[:, 550], dims), cmap='gray')
    # plt.show()
    #
    # plt.imshow(np.reshape(M[:, 140] - low_rank[:, 140], dims), cmap='gray')
    # plt.show()

    # m, n = M.shape
    # print(round(m*.05))

    L, S, examples = pcp(M, maxiter=5, k = 10)

    # plots(examples, dims, rows=5)

    f = plt_images(M, S, L, [140], dims)