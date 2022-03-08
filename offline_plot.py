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
    df = pd.read_csv('obs_and_states_29.csv', index_col=0)

    print(df)
    print(df.columns)

    time = df.index

    latent = df[[x for x in df.columns if 'z' in x]].iloc[:, :z_dim]
    ground = df[[x for x in df.columns if 'ground' in x]]
    obs = df[[x for x in df.columns if 'obs' in x]]
    obs_mu = obs[[x for x in obs.columns if 'scale' not in x]]
    obs_sigma = obs[[x for x in obs.columns if 'scale' in x]]

    n_obs = obs_mu.shape[1]

    # bodacious colors
    colors = sns.color_palette("rocket", 3)
    # Ram's colors, if desired
    seshadri = ['#c3121e', '#0348a1', '#ffb01c', '#027608', '#0193b0', '#9c5300', '#949c01', '#7104b5']
    #            0sangre,   1neptune,  2pumpkin,  3clover,   4denim,    5cocoa,    6cumin,    7berry

    # plot observations
    # plot
    fig = plt.figure(1, figsize=(20, 10))
    gs = gridspec.GridSpec(n_obs // 2, n_obs // 2)
    gs.update(wspace=0.2, hspace=0.25)  # spacing between subplots

    for i in range(n_obs):
        y1 = obs_mu['obs_{}'.format(i)]
        y2 = ground['ground_{}'.format(obs_idx[i])]
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
        plt.ylabel(r'position [m]', fontsize=14)  # label the y axis

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
    # TODO Needs beautification
    for i in range(z_dim // 2):
        xtr_subsplot = fig.add_subplot(gs[i])

        x, y = latent.iloc[:, i], latent.iloc[:, i+z_dim//2]
        x_g, y_g = ground.iloc[:, obs_idx[i]], ground.iloc[:, obs_idx[i+z_dim//2]]

        latent_mat = np.stack([x, y], axis=0)
        ground_mat = np.stack([x_g, y_g], axis=0)
        T, _, _, _ = np.linalg.lstsq(latent_mat, ground_mat, rcond=None)
        latent_rot = latent_mat @ T

        #plt.plot(x, y)
        plt.plot(x_g, y_g, color=colors[0])
        plt.scatter(latent_rot[0, :], latent_rot[1, :], color=colors[1])

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

        plt.plot(x_g, y_g, color=colors[0])
        plt.scatter(latent_rot[0, :], latent_rot[1, :], color=colors[1])
    plt.show()

    # plot energy
    # TODO Add energy to saved dataframe

    # plt.savefig('data_for_exercises/plotting/generic_plot.png', dpi=300,bbox_inches="tight")

    return 0


if __name__ == '__main__':

    offline_plot()
