import numpy as np

from .utils import gray_code, hadamard_matrix, get_mask
from .utils import validate_stats, visualize


def sequency_order(n):
    G = gray_code(n)
    G = G[:, ::-1]
    G = np.dot(G, 2 ** np.arange(G.shape[1] - 1, -1, -1)).astype(np.int32)
    return G


def get_matrix(n):
    H = hadamard_matrix(n)
    indexs = sequency_order(n)
    return H[indexs]


def get_index_matrix(n):
    indexs = np.arange(n**2)[::-1]
    indexs = indexs.reshape(n, n, order="F")

    index_matrix = indexs.astype(np.int32) + 1
    # reverse odd columns
    index_matrix[:, 1::2] = index_matrix[::-1, 1::2]

    return index_matrix


def main(n=64, m=256):
    H = get_matrix(n)
    index_matrix = get_index_matrix(n)
    mask = get_mask(index_matrix, n, m)

    validate_stats(index_matrix)
    visualize(H, index_matrix, mask)


if __name__ == "__main__":
    main()
