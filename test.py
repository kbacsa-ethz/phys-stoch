import os
import argparse
import configparser
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pyro
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    config_enumerate,
)
from pyro.optim import ClippedAdam

# visualization
from phys_data import TrajectoryDataset
from models import *
from dmm import DMM
from utils import data_path_from_config

import scipy.signal as signal
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from dynamics import *


def test(cfg):
    # add DLSC parameters like seed
    seed = 42
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    checkpoint = torch.load("checkpoints/duffing.pth")

    # use gpu
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    config = configparser.ConfigParser()
    config.read(os.path.join(cfg.root_path, cfg.config_path))

    exp_name = data_path_from_config(config)
    obs_idx = list(map(int, config['Simulation']['Observations'].split(',')))

    states = np.load(os.path.join(cfg.root_path, cfg.data_dir, exp_name, 'state.npy'))
    observations = np.load(os.path.join(cfg.root_path, cfg.data_dir, exp_name, 'obs.npy'))
    forces = np.load(os.path.join(cfg.data_dir, exp_name, 'force.npy'))
    energy = np.load(os.path.join(cfg.data_dir, exp_name, 'energy.npy'))

    # normalize
    obs_mean = observations.mean(axis=(0, 1), keepdims=True)
    obs_std = observations.std(axis=(0, 1), keepdims=True)
    states_mean = states.mean(axis=(0, 1), keepdims=True)
    states_std = states.std(axis=(0, 1), keepdims=True)
    observations_normalize = (observations - obs_mean) / obs_std
    states_normalize = (states - states_mean) / states_std

    # Save normalization parameters
    print("states_mean: {}".format(states_mean))
    print("states_std: {}".format(states_std))
    print("obs_mean: {}".format(obs_mean))
    print("obs_std: {}".format(obs_std))

    n_exp = states.shape[0]
    # 80/10/10 split by default
    indexes = np.arange(n_exp)
    train_idx = indexes[:int(0.8 * len(indexes))]
    val_idx = indexes[int(0.8 * len(indexes)):int(0.9 * len(indexes))]
    test_idx = indexes[int(0.9 * len(indexes)):]

    test_dataset = TrajectoryDataset(states[test_idx], observations[test_idx],
                                     forces[test_idx], obs_idx, states.shape[1])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # modules
    input_dim = len(obs_idx)
    z_dim = int(states.shape[-1] * 2 / 3)
    emitter = Emitter(input_dim, z_dim, cfg.emission_dim, cfg.emission_layers)
    transition = GatedTransition(z_dim, cfg.transmission_dim)
    if cfg.encoder_type == "birnn":
        combiner = CombinerBi(z_dim, z_dim)
    else:
        combiner = Combiner(z_dim, z_dim)

    if cfg.encoder_type == "rnn":
        encoder = RNNEncoder(input_dim, z_dim,
                             non_linearity='relu', batch_first=True,
                             num_layers=cfg.encoder_layers, dropout=cfg.encoder_dropout_rate)
    elif cfg.encoder_type == "birnn":
        encoder = BiRNNEncoder(input_dim, z_dim,
                               non_linearity='relu', batch_first=True,
                               num_layers=cfg.encoder_layers, dropout=cfg.encoder_dropout_rate)
    elif cfg.encoder_type == "node":
        encoder = ODEEncoder(input_dim, z_dim, cfg.potential_hidden, cfg.potential_layers,
                             non_linearity='relu', batch_first=True,
                             rnn_layers=cfg.encoder_layers, dropout=cfg.encoder_dropout_rate,
                             order=cfg.integrator_order,
                             dt=cfg.dt, discretization=cfg.discretization)
    elif cfg.encoder_type == "symplectic_node":
        encoder = SymplecticODEEncoder(input_dim, z_dim, cfg.potential_hidden, cfg.potential_layers,
                                       non_linearity='relu', batch_first=True,
                                       rnn_layers=cfg.encoder_layers, dropout=cfg.encoder_dropout_rate,
                                       order=cfg.integrator_order, dissipative=cfg.dissipative,
                                       learn_kinetic=cfg.learn_kinetic,
                                       dt=cfg.dt, discretization=cfg.discretization)
    else:
        raise NotImplementedError

    # create model
    vae = DMM(emitter, transition, combiner, encoder, z_dim,
              (cfg.encoder_layers, cfg.batch_size, z_dim))
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.to(device)

    # setup optimizer
    adam_params = {"lr": cfg.learning_rate,
                   "betas": (cfg.beta1, cfg.beta2),
                   "clip_norm": cfg.clip_norm, "lrd": cfg.lr_decay,
                   "weight_decay": cfg.weight_decay}
    adam = ClippedAdam(adam_params)
    elbo = Trace_ELBO()
    svi = SVI(vae.model, vae.guide, adam, loss=elbo)

    vae.eval()

    # TODO replace with itinialize simulation
    # parse system parameters

    print(cfg)

    system_type = config['System']['Name']
    flow_type = config['System']['Dynamics']
    m = np.diag(np.array(list(map(float, config['System']['M'].split(',')))))
    n_dof = m.shape[0]
    c = np.reshape(np.array(list(map(float, config['System']['C'].split(',')))), [n_dof, n_dof])
    k = np.reshape(np.array(list(map(float, config['System']['K'].split(',')))), [n_dof, n_dof])

    # parse external forces
    # TODO Add additional types of forces
    force_type = config['Forces']['Type']
    force_amp = float(config['Forces']['Amplitude'])
    force_freq = float(config['Forces']['Frequency'])
    force_shift = float(config['Forces']['Shift'])
    force_dof = np.array(list(map(int, config['Forces']['Inputs'].split(','))))

    # parse simulation parameters
    seed = int(config['Simulation']['Seed'])
    obs_idx = np.array(list(map(int, config['Simulation']['Observations'].split(','))))
    obs_noise = np.array(list(map(float, config['Simulation']['Noise'].split(','))))
    abserr = float(config['Simulation']['Absolute'])
    relerr = float(config['Simulation']['Relative'])
    dt = float(config['Simulation']['Delta'])
    t_max = float(config['Simulation']['Time'])

    # fix random seed for reproducibility
    np.random.seed(seed)

    tics = np.linspace(0., t_max, num=int(t_max / dt), endpoint=False)

    if force_type == 'free':
        force_fct = lambda x: 0
    elif force_type == 'impulse':
        force_fct = lambda x: signal.unit_impulse(len(tics), [x, x + 1, x + 3])
    elif force_type == 'sinusoidal':
        force_fct = np.sin
    else:
        raise NotImplementedError()

    if flow_type == 'linear':
        vectorfield = linear
    elif flow_type == 'duffing':
        vectorfield = duffing
    elif flow_type == 'pendulum':
        vectorfield, p = pendulum(n_dof)
    else:
        raise NotImplementedError()

    # run simulation
    q0_start = 0.5
    qdot0_start = 0.5
    q0, qdot0 = q0_start * np.ones([n_dof, 1]).squeeze(), qdot0_start * np.ones([n_dof, 1]).squeeze()
    w0 = np.concatenate([q0, qdot0], axis=0)

    # generate external forces
    force_input = np.zeros([n_dof, len(tics)])
    for dof in force_dof:
        if force_type == "impulse":
            impulse_shift = np.random.randint(0, len(tics) // 2)
            force_input[dof, :] = (force_amp * (0.25 + np.random.random())) * force_fct(impulse_shift)
        if force_type == "sinusoidal":
            force_input[dof, :] = (force_amp * np.random.random()) * force_fct(
                2 * np.pi * (force_freq * np.random.random()) * tics * dt)

    fint = interp1d(tics, force_input, fill_value='extrapolate')

    if flow_type == 'linear':
        p = [m, c, k, fint]
    if flow_type == 'duffing':
        p = [m, c, k, k / 3, fint]
    else:
        pass

    # Call the ODE solver.
    wsol = odeint(vectorfield, w0, tics, args=(p,),
                  atol=abserr, rtol=relerr)

    # recover acceleration
    wsol_dot = np.zeros_like(wsol)
    for idx, step in enumerate(tics):
        wsol_dot[idx, :] = vectorfield(wsol[idx, :], step, p)

    # join states and measure
    state = np.concatenate([wsol, wsol_dot[:, n_dof:]], axis=1)

    # calcuate energy of system
    q = state[:, :n_dof, None]
    qdot = state[:, n_dof:2 * n_dof, None]
    kinetic = 0.5 * np.matmul(np.transpose(qdot, axes=[0, 2, 1]), np.matmul(m, qdot))
    potential = 0.5 * np.matmul(np.transpose(q, axes=[0, 2, 1]), np.matmul(k, q))

    obs = np.zeros([state.shape[0], len(obs_idx)])
    for i, idx in enumerate(obs_idx):
        obs[:, i] = state[:, idx] + np.random.randn(state.shape[0]) * obs_noise[i]

    n_len = cfg.seq_len * 10

    n_obs = 4
    dt = cfg.dt
    time = np.arange(0, n_len * dt, dt, dtype=float)
    obs = (obs - obs_mean) / obs_std

    sample_obs = torch.from_numpy(obs[:, :n_len + 1, :]).float()
    sample_obs = sample_obs.to(device)
    Z, Z_gen, Z_gen_scale, Obs, Obs_scale = vae.reconstruction(sample_obs)
    Obs = Obs.detach() * obs_std + obs_mean
    Obs_scale = Obs_scale.detach() * obs_std + obs_mean
    ground_truth = torch.from_numpy(state[:n_len, obs_idx]).unsqueeze(0).float()
    ground_truth = ground_truth.to(device)
    mse = torch.abs((Obs - ground_truth) / ground_truth).mean().item()
    error = torch.logical_or(torch.lt(ground_truth, (Obs - 2 * Obs_scale)),
                             torch.gt(ground_truth, (Obs + 2 * Obs_scale))).float().mean().item()
    t_vec = torch.arange(1, n_len + 1) * dt

    q = Z[:, :, :z_dim // 2]
    qd = Z[:, :, z_dim // 2:]
    qdot = qd
    qdot = qdot[..., None].squeeze(0).detach().numpy()

    if cfg.dissipative:
        input_tensor = torch.cat([t_vec.float().unsqueeze(1), q], dim=1)
    else:
        # input_tensor = torch.cat([torch.from_numpy(q).float(), torch.from_numpy(qd).float()], dim=1)
        input_tensor = q

    latent_potential = vae.encoder.latent_func.energy(t_vec, input_tensor).detach().numpy()

    m = np.eye(z_dim // 2)
    latent_kinetic = 0.5 * np.matmul(np.transpose(qdot, axes=[0, 2, 1]), np.matmul(m, qdot))
    latent_kinetic = latent_kinetic.flatten()

    print(mse)
    print(error)
    colors = sns.color_palette("viridis", 4)

    # normalize energy
    latent_potential = latent_potential[:, 5:, :]
    latent_kinetic = latent_kinetic[5:]

    latent_kinetic = (latent_kinetic - latent_kinetic.min()) / (latent_kinetic.max() - latent_kinetic.min())
    latent_potential = (latent_potential - latent_potential.min()) / (latent_potential.max() - latent_potential.min())
    kinetic = kinetic[5:n_len]
    potential = potential[5:n_len]

    kinetic = (kinetic - kinetic.min()) / (kinetic.max() - kinetic.min())
    potential = (potential - potential.min()) / (potential.max() - potential.min())

    fig = plt.figure(1, figsize=(20, 10))
    gs = gridspec.GridSpec(n_obs // 2, n_obs // 2)
    gs.update(wspace=0.2, hspace=0.25)  # spacing between subplots
    for i in range(n_obs):
        y1 = Obs[:, :, i].squeeze().detach().numpy()
        y2 = state[:n_len, obs_idx[i]]
        sample_y = sample_obs[:, :n_len, i].squeeze().detach().numpy() * obs_std[:, :, i] + obs_mean[:, :, i]
        bound = Obs_scale[:, :, i].squeeze().detach().numpy()
        lower_bound = y1 - bound

        # plt.plot(time, y1)
        # plt.plot(time, y2)
        # plt.fill_between(time, y1-bound, y1+bound, facecolor='yellow', alpha=0.3)
        # plt.show()

        xtr_subsplot = fig.add_subplot(gs[i])

        plt.plot(time, y1, linestyle='-', label='predicted mean', color=colors[0])  # plot data
        plt.plot(time, y2, linestyle='-', label='ground truth', color=colors[1])  # plot data
        plt.fill_between(time, y1 - 2 * bound, y1 + 2 * bound, facecolor=colors[2], alpha=0.3,
                         label=r'2-$\sigma$ range')
        plt.scatter(time, sample_y, s=2, color=colors[3], label='noisy samples')

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
    # plt.show()

    # plot phase
    fig = plt.figure(2, figsize=(20, 10))
    gs = gridspec.GridSpec(z_dim // 2, z_dim // 2)

    # normalize and remove starting points
    latent = Z[:, 5:n_len, :].squeeze().detach().numpy()
    ground = state[5:n_len, :]
    latent = (latent - latent.min()) / (latent.max() - latent.min())
    ground = (ground - ground.min()) / (ground.max() - ground.min())

    # regular phase
    for i in range(z_dim // 2):
        xtr_subsplot = fig.add_subplot(gs[i])

        x, y = latent[:, i], latent[:, i + z_dim // 2]
        x_g, y_g = ground[:, obs_idx[i]], ground[:, obs_idx[i + z_dim // 2]]

        latent_mat = np.stack([x, y], axis=0)
        ground_mat = np.stack([x_g, y_g], axis=0)
        T, _, _, _ = np.linalg.lstsq(latent_mat, ground_mat, rcond=None)
        latent_rot = latent_mat @ T

        plt.xlabel(r'$x_1$', fontsize=14)
        plt.ylabel(r'$\dot{x}_1$', fontsize=14)
        plt.plot(x_g, y_g, color=colors[0], linestyle='dashed', linewidth=0.5, label='original')
        plt.scatter(latent_rot[0, :], latent_rot[1, :], color=colors[-1], label='predicted (rotated)')
        plt.legend(fontsize=14, loc='upper right')  # add the legend (will default to 'best' location)

    # TODO Automate this
    latent_swap = latent[:, [0, 1, 3, 2]]
    for i in range(z_dim // 2):
        xtr_subsplot = fig.add_subplot(gs[i + z_dim // 2])

        x, y = latent_swap[:, i], latent_swap[:, i + z_dim // 2]
        x_g, y_g = ground[:, obs_idx[i]], ground[:, obs_idx[i + z_dim // 2]]

        latent_mat = np.stack([x, y], axis=0)
        ground_mat = np.stack([x_g, y_g], axis=0)
        T, _, _, _ = np.linalg.lstsq(latent_mat, ground_mat, rcond=None)
        latent_rot = latent_mat @ T

        plt.plot(x_g, y_g, color=colors[0], linestyle='dashed', linewidth=0.5, label='original')
        plt.scatter(latent_rot[0, :], latent_rot[1, :], color=colors[-1], label='predicted (rotated)')
        plt.legend(fontsize=14, loc='upper right')  # add the legend (will default to 'best' location)

    fig = plt.figure(3)
    plt.plot(time[5:], latent_kinetic.squeeze(), color=colors[0])
    plt.plot(time[5:], latent_potential.squeeze(), color=colors[1])
    plt.plot(time[5:], kinetic.squeeze(), color=colors[2])
    plt.plot(time[5:], potential.squeeze(), color=colors[3])
    plt.show()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--config-path', type=str, default='config/2springmass_duffing_free_free.ini')
    parser.add_argument('-e', '--emission-dim', type=int, default=14)
    parser.add_argument('-ne', '--emission-layers', type=int, default=1)
    parser.add_argument('-tr', '--transmission-dim', type=int, default=36)
    parser.add_argument('-ph', '--potential-hidden', type=int, default=54)
    parser.add_argument('-pl', '--potential-layers', type=int, default=0)
    parser.add_argument('-tenc', '--encoder-type', type=str, default="symplectic_node")
    parser.add_argument('-nenc', '--encoder-layers', type=int, default=1)
    parser.add_argument('-ord', '--integrator-order', type=int, default=2)
    parser.add_argument('--dissipative', action='store_true')
    parser.add_argument('-dt', '--dt', type=float, default=0.1)
    parser.add_argument('-disc', '--discretization', type=int, default=3)
    parser.add_argument('-n', '--num-epochs', type=int, default=1)
    parser.add_argument('-te', '--tuning-epochs', type=int, default=10)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('-b1', '--beta1', type=float, default=0.96)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip-norm', type=float, default=10.0)
    parser.add_argument('-lrd', '--lr-decay', type=float, default=0.99996)
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.01)
    parser.add_argument('-bs', '--batch-size', type=int, default=256)
    parser.add_argument('-sq', '--seq-len', type=int, default=50)
    parser.add_argument('-ae', '--annealing-epochs', type=int, default=2)
    parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.5)
    parser.add_argument('-rdr', '--encoder-dropout-rate', type=float, default=0.1)
    parser.add_argument('-iafs', '--num-iafs', type=int, default=0)
    parser.add_argument('-id', '--iaf-dim', type=int, default=100)
    parser.add_argument('-vf', '--validation-freq', type=int, default=1)
    parser.add_argument('--learn-kinetic', action='store_true')
    parser.add_argument('--nproc', type=int, default=2)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--comet', action='store_true')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    test(args)
