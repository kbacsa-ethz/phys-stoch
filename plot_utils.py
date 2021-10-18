import matplotlib.pyplot as plt
import numpy as np
from utils import normalize
from rotation import rigid_transform_2D
from MEMD_all import memd


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


def phase_plot(pred_pos, pred_vec, grnd_pos, grnd_vec, title, debug=False):

    n_plots = pred_pos.shape[-1]

    fig1 = plt.figure(figsize=(16, 7))
    plt.title(title)
    for i in range(n_plots):
        ax = plt.subplot(n_plots // 2, n_plots // (n_plots // 2), i + 1)

        x1 = normalize(pred_pos[2:, i])
        x1d = normalize(pred_vec[2:, i])
        y1 = normalize(grnd_pos[2:, i])
        y1d = normalize(grnd_vec[2:, i])

        plt.xlabel("x_{}".format(i))
        plt.ylabel("xdot_{}".format(i))
        plt.plot(x1, x1d, '--', label='latent phase')
        plt.plot(y1, y1d, label='true phase')
        plt.legend(loc="upper left")

    fig2 = plt.figure(figsize=(16, 7))
    plt.title(title)
    t_max = 450
    for i in range(n_plots//2):
        ax = plt.subplot(n_plots // 2, n_plots // (n_plots // 2), i + 1)

        x1 = normalize(pred_pos[2:t_max, i])
        x1d = normalize(pred_pos[2:t_max, i+1])
        y1 = normalize(grnd_pos[2:t_max, i])
        y1d = normalize(grnd_pos[2:t_max, i+1])

        rot_out = rigid_transform_2D(
            np.stack([x1, x1d], axis=0),
            np.stack([y1, y1d], axis=0)
        )

        xr, yr = rot_out[0, :], rot_out[1, :]

        plt.xlabel("x_{}".format(i))
        plt.ylabel("x_{}".format(i+1))
        plt.plot(x1, x1d, '--', label='latent phase')
        plt.plot(y1, y1d, label='true phase')
        plt.plot(xr, yr, label='rotated phase')
        plt.legend(loc="upper left")

        ax = plt.subplot(n_plots // 2, n_plots // (n_plots // 2), i + 2)

        x1 = normalize(pred_vec[2:t_max, i])
        x1d = normalize(pred_vec[2:t_max, i+1])
        y1 = normalize(grnd_vec[2:t_max, i])
        y1d = normalize(grnd_vec[2:t_max, i+1])

        rot_out = rigid_transform_2D(
            np.stack([x1, x1d], axis=0),
            np.stack([y1, y1d], axis=0)
        )

        xr, yr = rot_out[0, :], rot_out[1, :]

        plt.xlabel("xdot_{}".format(i))
        plt.ylabel("xdot_{}".format(i+1))
        plt.plot(x1, x1d, '--', label='latent phase')
        plt.plot(y1, y1d, label='true phase')
        plt.plot(xr, yr, label='rotated phase')
        plt.legend(loc="upper left")

    if debug:
        plt.show()
    return fig1, fig2


def plot_emd(state_space, debug=False):
    imf = memd(state_space)
    M, D, T = imf.shape
    fig, axes = plt.subplots(nrows=D, ncols=M, figsize=(30, 15))
    for i in range(D):
        for j in range(M):
            ax = plt.subplot(D, M, M * i + j + 1)
            plt.plot(imf[j, i, :])

    cols = ['Mode {}'.format(col) for col in range(M)]
    rows = ['State {}'.format(row) for row in range(D)]

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=0, size='large')

    fig.tight_layout()

    if debug:
        plt.show()
    return fig
