import numpy as np


def hadamard_matrix(n):
    if n == 1:
        return np.array([[1]])
    else:
        h = hadamard_matrix(n // 2)
        return np.block([[h, h], [h, -h]])


def gray_code(n):
    g0 = np.array([[0], [1]])
    g = g0

    while g.shape[0] < n:
        g = np.hstack(
            [np.kron(g0, np.ones((g.shape[0], 1))), np.vstack([g, g[::-1, :]])]
        )
    return g


def get_order(input):
    h, w = input.shape
    indx = np.argsort(input.reshape(-1))
    values = np.arange(0, h * w)
    values[indx] = values.copy()
    values = values.reshape(h, w) + 1
    return values


def get_coords(n):
    X = np.arange(n)
    X, Y = np.meshgrid(X, X)
    return X, Y


def get_coords_array(n):
    coords = np.stack(get_coords(n), -1)
    return coords


def coord_ordering(n, get_index_matrix=None):
    coords = get_coords_array(n).reshape(-1, 2)
    index_matrix = get_index_matrix(n).reshape(-1)
    # order by sum of cords in descending order
    order = np.argsort(index_matrix)[::-1]
    coords = coords[order]

    return order, coords


def get_mask(index_matrix, n, m=1):
    mask = index_matrix > n**2 - m
    return mask


def get_prob_mask(index_matrix, n, sr):
    index_matrix = index_matrix.max() - index_matrix + 1
    prob = sr ** ((index_matrix - 1) / n**2)

    plt.imshow(prob, cmap="jet")
    plt.show()

    prob_mask = np.random.binomial(1, prob)
    return prob_mask


"""
    ---------------- plot functions ----------------
"""

import matplotlib.pyplot as plt


def validate_stats(index_matrix):
    # perform range, dtype, and shape checks in the index matrix

    print("index matrix shape: ", index_matrix.shape)
    print("index matrix dtype: ", index_matrix.dtype)
    print("index matrix min: ", index_matrix.min())
    print("index matrix max: ", index_matrix.max())


def visualize(H, index_matrix, mask):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(H, cmap="gray")
    plt.xticks([])
    plt.yticks([])

    plt.title("hadamard matrix")

    plt.subplot(1, 3, 2)
    plt.imshow(index_matrix, cmap="jet")
    plt.xticks([])
    plt.yticks([])
    plt.title("index matrix")

    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.title("mask")
    plt.show()
