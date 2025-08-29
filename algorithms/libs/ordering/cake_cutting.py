import numpy as np

from .utils import get_mask
from .utils import validate_stats, visualize

from .sequency import (
    get_matrix as hadamard_matrix,
    get_index_matrix as sequency_index_matrix,
)


def cake_cutting_seq(i, p):
    """Sequence of i-th"""
    step = -i * (-1) ** (np.mod(i, 2))

    seq = None
    # if i is odd
    if np.mod(i, 2) == 1:
        seq = list(range(i, i * p + 1, step))
    else:
        seq = list(range(i * p, i - 1, step))

    return seq


def cake_cutting_order(n):
    """Cake cutting order"""
    p = int(np.sqrt(n))
    seq = [cake_cutting_seq(i, p) for i in range(1, p + 1)]
    seq = [item for sublist in seq for item in sublist]
    return np.argsort(seq)


def get_index_matrix(n):
    indexs = sequency_index_matrix(n).flatten()
    cake_order = cake_cutting_order(n**2)

    cake_order = cake_order.reshape(n, n, order="F")
    cake_order[:, 1::2] = cake_order[::-1, 1::2]
    cake_order = cake_order.flatten()

    indexs[cake_order] = indexs.copy()
    indexs = indexs.reshape(n, n)
    indexs[1::2, :] = indexs[1::2, ::-1]
    indexs = indexs.T
    return indexs


def get_matrix(n):
    H = hadamard_matrix(n)
    indexs = cake_cutting_order(n)
    return H[indexs]


def main(n=64, m=256):
    H = get_matrix(n)
    index_matrix = get_index_matrix(n)
    mask = get_mask(index_matrix, n, m)

    validate_stats(index_matrix)
    visualize(H, index_matrix, mask)


if __name__ == "__main__":
    main()
