import matplotlib.pyplot as plt
import numpy as np
from utils import normalize


def simple_plot(x_axis, values, max_t, names, title, normalized=False, debug=False):
    assert max_t < len(x_axis)
    fig = plt.figure(figsize=(16, 7))
    plt.title(title)
    for v, n in zip(values, names):
        if normalized:
            v = normalize(v)
        plt.plot(x_axis[:max_t], v[:max_t], label=n)
    plt.xlabel("$t$")
    plt.legend(loc="upper left")
    if debug:
        plt.show()
    return fig


def grid_plot(x_axis, values, uncertainty, max_1, max_2, n_plots, names, title, y_label, debug=False):
    fig = plt.figure(figsize=(16, 7))
    plt.title(title)
    for i in range(n_plots):
        ax = plt.subplot(n_plots // 2, n_plots // (n_plots // 2), i + 1)
        for v, u, n in zip(values, uncertainty, names):
            plt.plot(x_axis, v[max_1, :max_2, i], label=n)  # plot first instance as default
            if u is not None:
                lower_bound = v[max_1, :max_2, i] - u[max_1, :max_2, i]
                upper_bound = v[max_1, :max_2, i] + u[max_1, :max_2, i]
                ax.fill_between(x_axis, lower_bound, upper_bound,
                                facecolor='yellow', alpha=0.5,
                                label='1 sigma range')

        plt.legend(loc="upper left")
        plt.xlabel("$t$")
        plt.ylabel(y_label[i])
    fig.suptitle(title)
    plt.tight_layout()
    if debug:
        plt.show()
    return fig


def matrix_plot(matrix, title, debug=False):
    fig = plt.figure(figsize=(16, 7))
    plt.title(title)
    plt.imshow(matrix)
    for i in np.arange(np.shape(matrix)[0]):  # over all rows of count
        for j in np.arange(np.shape(matrix)[1]):  # over all cols of count
            text = plt.text(j, i, str(round(matrix[i, j], 2)), ha="center", va="center", color="r")
    if debug:
        plt.show()
    return fig
