import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec


plt.rc('font', family='serif', size=13)

Obs = np.load('intermediate/Obs.npy')
Z_gen = np.load('intermediate/Z_gen.npy')
Obs_scale = np.load('intermediate/Obs_scale.npy')
state = np.load('intermediate/state.npy')
sample_obs = np.load('intermediate/sample_obs.npy')

latent_kinetic = np.load('intermediate/latent_kinetic.npy')
latent_potential = np.load('intermediate/latent_potential.npy')
kinetic = np.load('intermediate/kinetic.npy')
potential = np.load('intermediate/potential.npy')

obs_list = [0, 1, 4, 5]

print(Obs.shape)
print(Z_gen.shape)
print(Obs_scale.shape)
print(state.shape)
print(sample_obs.shape)

n_len = 500
dt = 0.1
n_dof = 2
t = np.arange(0, n_len * dt, dt, dtype=float)
colors = sns.color_palette("tab10", 6)
#colors = ['blue', 'green', 'yellow', 'red']

fig = plt.figure(1, figsize=(15, 7))
latent = Z_gen[0, 5:n_len, :]
ground = state[5:n_len, :]

x, y = latent[:, 0], latent[:, 2]
x_g, y_g = ground[:, 0], ground[:, 1]

x = (x - x.min()) / (x.max() - x.min())
y = (y - y.min()) / (y.max() - y.min())
x_g = (x_g - x_g.min()) / (x_g.max() - x_g.min())
y_g = (y_g - y_g.min()) / (y_g.max() - y_g.min())

latent_mat = np.stack([x, y], axis=0)
ground_mat = np.stack([x_g, y_g], axis=0)
T, _, _, _ = np.linalg.lstsq(latent_mat, ground_mat, rcond=None)
latent_rot = latent_mat @ T
latent_rot = latent_rot.T
ground_mat = ground_mat.T

fig = plt.figure(1, figsize=(15, 7))
plt.plot(ground_mat[:, 0], ground_mat[:, 1], linestyle='dashed', linewidth=1.5, label='true phase')
plt.scatter(latent_rot[:, 0], latent_rot[:, 1], color='r', label='latent phase (rotated)')
plt.legend(loc='upper left')
plt.xlabel(r'$x_0$ (normalized)')
plt.ylabel(r'$x_1$ (normalized)')
plt.show()

fig = plt.figure(1, figsize=(15, 7))
gs = gridspec.GridSpec(2, 2)
gs.update(wspace=0.15, hspace=0.35)  # spacing between subplots

"""
titles = [
    r"$z_1$ (front suspension)",
    r"$z_2$ (axle translation)",
    r"$z_3$ (axle rotation)",
    r"$z_4$ (rear suspension)",
    r"$z_5$ (front seat)",
    r"$z_6$ (rear seat)",
]
"""

titles = [
    r"$x_0(t)$",
    r"$x_1(t)$",
    r"$\ddot{x}_0(t)$",
    r"$\ddot{x}_1(t)$",
]

for idx, obs_idx in enumerate(obs_list):
    y1 = state[:n_len, obs_idx]
    y2 = Obs[0, :n_len, idx]
    sample_y = sample_obs[0, :n_len, idx]
    bound = Obs_scale[0, :n_len, idx]

    xtr_subsplot = fig.add_subplot(gs[idx])
    plt.plot(t, y1, linestyle='-', label='ground truth', color=colors[0])  # plot data
    plt.plot(t, y2, linestyle='-', label='predicted mean', color=colors[1])  # plot data
    plt.fill_between(t, y2 - 2 * bound, y2 + 2 * bound, facecolor=colors[2], alpha=0.5,
                     label=r'2-$\sigma$ range')
    plt.scatter(t, sample_y, s=2, color=colors[3], label='noisy samples')
    plt.legend(fontsize=10, loc='upper right')  # add the legend (will default to 'best' location)
    plt.xlabel('time [s]')

    if idx > 0:
        plt.ylabel(r"acceleration [m/$s^2$]")
    else:
        plt.ylabel('displacement [m]')
    plt.title(titles[idx])
    plt.xlim([0, n_len*dt])
plt.show()

fig = plt.figure(3)
max_iter = 500
plt.plot(t[:max_iter], latent_kinetic.squeeze()[:max_iter], color=colors[0], label='learned kinetic')
plt.plot(t[:max_iter], latent_potential.squeeze()[:max_iter], color=colors[1], label='learned potential')
plt.plot(t[:max_iter], kinetic.squeeze()[:max_iter], color=colors[-1], label='true kinetic')
plt.plot(t[:max_iter], potential.squeeze()[:max_iter], color=colors[-2], label='true potential')
learned_mechanical = latent_kinetic.squeeze()[:max_iter] + latent_potential.squeeze()[:max_iter]
mechanical = kinetic.squeeze()[:max_iter] + potential.squeeze()[:max_iter]

plt.plot(t[:max_iter], learned_mechanical, color=colors[2], linestyle='--', label='learned mechanical')
plt.plot(t[:max_iter], mechanical, color=colors[-3], linestyle='--', label='true mechanical')

plt.xlabel(r'time [s]', fontsize=14)
plt.ylabel('Energy [J]', fontsize=14)  # label the y axis
plt.legend(fontsize=11, loc='upper right')  # add the legend (will default to 'best' location)
plt.xlim([0, 10.])
plt.ylim([0, 1.75])
plt.show()
