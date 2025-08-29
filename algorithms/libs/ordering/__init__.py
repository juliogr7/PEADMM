from .hadamard import (
    get_matrix as hadamard_matrix,
    get_index_matrix as hadamard_index_matrix,
)
from .sequency import (
    get_matrix as sequency_matrix,
    get_index_matrix as sequency_index_matrix,
)
from .cake_cutting import (
    get_matrix as cake_cutting_matrix,
    get_index_matrix as cake_cutting_index_matrix,
)
from .zig_zag import (
    get_matrix as zig_zag_matrix,
    get_index_matrix as zig_zag_index_matrix,
)
from .xy import get_matrix as xy_matrix, get_index_matrix as xy_index_matrix

from .utils import get_mask


MATRIX_FUNCTIONS = {
    "hadamard": hadamard_matrix,
    "sequency": sequency_matrix,
    "cake_cutting": cake_cutting_matrix,
    "zig_zag": zig_zag_matrix,
    "XY": xy_matrix,
}

INDEX_MATRIX_FUNCTIONS = {
    "hadamard": hadamard_index_matrix,
    "sequency": sequency_index_matrix,
    "cake_cutting": cake_cutting_index_matrix,
    "zig_zag": zig_zag_index_matrix,
    "XY": xy_index_matrix,
}


def get_matrix(n, ordering="hadamard"):
    return MATRIX_FUNCTIONS[ordering](n)


def get_index_matrix(n, ordering="hadamard"):
    return INDEX_MATRIX_FUNCTIONS[ordering](n)


def get_n_mask(size, n, ordering="hadamard"):
    index_matrix = get_index_matrix(size, ordering)
    mask = get_mask(index_matrix, size, n)
    return mask


def plot(n=64):
    import matplotlib.pyplot as plt

    orderings = ["XY"]
    # orderings = ["sequency", "zig_zag", "cake_cutting", "XY"]

    plt.figure(figsize=(8, 6))
    m = n * 4

    for i, ordering in enumerate(orderings):
        H = get_matrix(n, ordering)
        index_matrix = get_index_matrix(n, ordering)
        mask = get_mask(index_matrix, n, m)

        plt.subplot(3, len(orderings), i + 1)
        plt.imshow(H, cmap="gray")
        plt.xticks([])
        plt.yticks([])

        # make title, capitalize first letter and remove "_"
        ordering = ordering.replace("_", " ")
        ordering = ordering[0].upper() + ordering[1:]
        plt.title(ordering)

        plt.subplot(3, len(orderings), len(orderings) + 1 + i)
        plt.imshow(index_matrix, cmap="jet")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, len(orderings), len(orderings) * 2 + 1 + i)
        plt.imshow(mask, cmap="gray")
        plt.xticks([])
        plt.yticks([])

    # reduce space between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig("ordering.pdf", dpi=300, bbox_inches="tight")


def speed_test():
    import matplotlib.pyplot as plt
    import time
    import numpy as np
    import pandas as pd
    import seaborn as sns

    orderings = ["sequency", "zig_zag", "cake_cutting", "XY"]

    trials = 100
    n_values = [64, 128, 256, 512]

    df = pd.DataFrame(columns=["n", "ordering", "time"])

    print("Running speed test...")
    for n in n_values:
        print(f"n = {n}")
        for ordering in orderings:
            print(f"    {ordering}")
            for _ in range(trials):
                start_time = time.time()
                _ = get_index_matrix(n, ordering)
                end_time = time.time()
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "n": n,
                                "ordering": ordering,
                                "time": end_time - start_time,
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )

    print("Plotting...")

    # boxplot n vs time and hue for ordering
    sns.boxplot(x="n", y="time", hue="ordering", data=df)

    plt.show()


if __name__ == "__main__":
    plot()
