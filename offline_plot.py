import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
import seaborn as sns

import matplotlib.font_manager

fpaths = matplotlib.font_manager.findSystemFonts()

"""
for i in fpaths:
    f = matplotlib.font_manager.get_font(i)
    print(f.family_name)
"""

rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
# rc('text', usetex=True)


def offline_plot():

    # TODO How can this be passed automatically ?
    obs_idx = [0, 1, 4, 5]
    z_dim = 4

    case = 'duffing'

    if case == 'duffing':
        obs_mean = np.array([[[-0.00066353, -0.0009563, 0.00039857, 0.00119788]]])
        obs_std = np.array([[[0.94974715, 0.95087484, 5.02525429, 5.98210665]]])
    if case == 'pendulum':
        obs_mean = np.array([[[0.00423583, -0.02556212, -0.00433946, -0.00444246]]])
        obs_std = np.array([[[0.54962711, 1.61728878, 0.8287365, 1.092226]]])

    df = pd.read_csv('plot_data/{}.csv'.format(case), index_col=0)

    print(df)
    print(df.columns)

    time = df.index

    latent = df[[x for x in df.columns if 'z' in x]].iloc[:, :z_dim]
    ground = df[[x for x in df.columns if 'ground' in x]]
    obs = df[[x for x in df.columns if 'obs' in x]]
    obs_mu = obs[[x for x in obs.columns if 'scale' not in x]]
    obs_sigma = obs[[x for x in obs.columns if 'scale' in x]]
    sample = df[[x for x in df.columns if 'sample' in x]]

    n_obs = obs_mu.shape[1]

    sample = sample.mul(obs_std.squeeze(), axis=1)
    sample = sample.add(obs_mean.squeeze(), axis=1)

    # bodacious colors
    colors = sns.color_palette("viridis", 4)
    # Ram's colors, if desired

    #colors = ["#00ffb6", "#a6ffe6", "#ababab", "#00ffc9"]
    #colors = ['#c3121e', '#0348a1', '#ffb01c', '#027608', '#0193b0', '#9c5300', '#949c01', '#7104b5']
    #            0sangre,   1neptune,  2pumpkin,  3clover,   4denim,    5cocoa,    6cumin,    7berry

    # plot observations
    # plot
    fig = plt.figure(1, figsize=(20, 10))
    gs = gridspec.GridSpec(n_obs // 2, n_obs // 2)
    gs.update(wspace=0.2, hspace=0.25)  # spacing between subplots

    for i in range(n_obs):
        y1 = obs_mu['obs_{}'.format(i)]
        y2 = ground['ground_{}'.format(obs_idx[i])]
        sample_y = sample['sample_{}'.format(i)]
        bound = obs_sigma['obs_scale_{}'.format(i)]
        lower_bound = y1 - bound

        # plt.plot(time, y1)
        # plt.plot(time, y2)
        # plt.fill_between(time, y1-bound, y1+bound, facecolor='yellow', alpha=0.3)
        # plt.show()

        xtr_subsplot = fig.add_subplot(gs[i])

        plt.plot(time, y1, linestyle='-', label='predicted mean', color=colors[0])  # plot data
        plt.plot(time, y2, linestyle='-', label='ground truth', color=colors[1])  # plot data
        plt.fill_between(time, y1 - 2 * bound, y1 + 2 * bound, facecolor=colors[2], alpha=0.3, label=r'2-$\sigma$ range')
        plt.scatter(time, sample_y, s=2, c=colors[3], label='noisy samples')

        # plot params
        plt.xlim([min(time), max(time)])
        # plt.ylim([-0.5,16])
        plt.minorticks_on()
        plt.tick_params(direction='in', right=True, top=True)
        plt.tick_params(labelsize=14)
        plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
        xticks = np.arange(min(time), max(time), 5)
        yticks = np.arange(min(y1), max(y1), 0.5)

        plt.tick_params(direction='in', which='minor', length=5, bottom=True, top=True, left=True, right=True)
        plt.tick_params(direction='in', which='major', length=10, bottom=True, top=True, left=True, right=True)
        # plt.xticks(xticks)
        # plt.yticks(yticks)

        plt.xlabel(r'time [s]', fontsize=14)
        plt.ylabel(r'latent', fontsize=14)  # label the y axis

        plt.legend(fontsize=14, loc='upper right')  # add the legend (will default to 'best' location)
    #plt.show()

    # plot phase
    fig = plt.figure(2, figsize=(20, 10))
    gs = gridspec.GridSpec(z_dim // 2, z_dim // 2)

    # normalize
    latent = latent[5:]  # remove approximate first points
    ground = ground[5:]  # remove approximate first points
    latent = (latent - latent.min()) / (latent.max() - latent.min())
    ground = (ground - ground.min()) / (ground.max() - ground.min())

    # regular phase
    for i in range(z_dim // 2):
        xtr_subsplot = fig.add_subplot(gs[i])

        x, y = latent.iloc[:, i], latent.iloc[:, i+z_dim//2]
        x_g, y_g = ground.iloc[:, obs_idx[i]], ground.iloc[:, obs_idx[i+z_dim//2]]

        latent_mat = np.stack([x, y], axis=0)
        ground_mat = np.stack([x_g, y_g], axis=0)
        T, _, _, _ = np.linalg.lstsq(latent_mat, ground_mat, rcond=None)
        latent_rot = latent_mat @ T

        plt.xlabel(r'$x_1$', fontsize=14)
        plt.ylabel(r'$\dot{x}_1$', fontsize=14)
        plt.plot(x_g, y_g, color=colors[0], linestyle='dashed',  linewidth=0.5, label='original')
        plt.scatter(latent_rot[0, :], latent_rot[1, :], color=colors[-1], label='predicted (rotated)')
        plt.legend(fontsize=14, loc='upper right')  # add the legend (will default to 'best' location)

    # TODO Automate this
    latent_swap = latent.reindex(columns=['z_0', 'z_1', 'z_3', 'z_2'])
    for i in range(z_dim // 2):
        xtr_subsplot = fig.add_subplot(gs[i+z_dim//2])

        x, y = latent_swap.iloc[:, i], latent_swap.iloc[:, i + z_dim // 2]
        x_g, y_g = ground.iloc[:, obs_idx[i]], ground.iloc[:, obs_idx[i + z_dim // 2]]

        latent_mat = np.stack([x, y], axis=0)
        ground_mat = np.stack([x_g, y_g], axis=0)
        T, _, _, _ = np.linalg.lstsq(latent_mat, ground_mat, rcond=None)
        latent_rot = latent_mat @ T

        plt.plot(x_g, y_g, color=colors[0], linestyle='dashed', linewidth=0.5, label='original')
        plt.scatter(latent_rot[0, :], latent_rot[1, :], color=colors[-1], label='predicted (rotated)')
        plt.legend(fontsize=14, loc='upper right')  # add the legend (will default to 'best' location)

    plt.show()
    # plt.savefig('data_for_exercises/plotting/generic_plot.png', dpi=300,bbox_inches="tight")
    return 0


if __name__ == '__main__':

    offline_plot()
