"""
Row-wise ordering of a hadamard matrix.
"""
import numpy as np

H = np.array([[1, 1], [1, -1]])

def dec2bit(dec, n):
    bit = np.zeros(n)
    for i in range(n):
        bit[i] = np.array(dec % 2)
        dec = dec // 2
    return bit


def hadamard_row(row, n):
    bits = dec2bit(row, n)
    h = np.ones((1, 1))

    for i in range(n):
        hi = H[bits[i].astype(int)]
        h = np.kron(hi, h)
    return h