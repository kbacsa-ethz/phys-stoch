import matplotlib.pyplot as plt
import numpy as np


def simple_plot(x_axis, values, names, title, debug=False):
    fig = plt.figure(figsize=(16, 7))
    plt.title(title)
    for v, n in zip(values, names):
        plt.plot(x_axis, v, label=n)
    plt.xlabel("$t$")
    plt.legend(loc="upper left")
    if debug:
        plt.show()
    return fig


def grid_plot(x_axis, values, max_1, max_2, n_plots, names, title, y_label, debug=False):
    fig = plt.figure(figsize=(16, 7))
    plt.title(title)
    for i in range(n_plots):
        ax = plt.subplot(n_plots // 2, n_plots // (n_plots // 2), i + 1)
        for v, n in zip(values, names):
            plt.plot(x_axis, v[max_1, :max_2, i], label=n)  # plot first instance as default

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
