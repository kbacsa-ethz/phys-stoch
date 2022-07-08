import argparse
import configparser
import os
import json
import random
import numpy as np
import torch
import pyro
from models import *
from dynamics import *
from dmm import DMM
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import scipy.signal
import matplotlib as mpl


def test_model(cfg):
    # add DLSC parameters like seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    # use gpu
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    if cfg.headless:
        mpl.use('Agg')  # if you are on a headless machine
    else:
        mpl.use('TkAgg')

    ckpt_folder = os.listdir(os.path.join(cfg.root_path, 'experiments', cfg.ckpt_path))[-1]

    f = open(os.path.join(cfg.root_path, 'experiments', cfg.ckpt_path, ckpt_folder, 'config.txt'))
    model_parameters = json.load(f)
    checkpoint = torch.load(os.path.join(cfg.root_path, 'experiments', cfg.ckpt_path, ckpt_folder, 'checkpoint.pth'))

    config_path = model_parameters['config_path']

    config = configparser.ConfigParser()
    config.read(os.path.join(cfg.root_path, config_path))

    m = np.diag(np.array(list(map(float, config['System']['M'].split(',')))))
    obs_idx = list(map(int, config['Simulation']['Observations'].split(',')))
    n_dof = m.shape[0]

    # modules
    input_dim = len(obs_idx)
    z_dim = n_dof * 2

    combiner = Combiner(z_dim, z_dim)
    emitter = Emitter(input_dim, z_dim, model_parameters['emission_dim'], model_parameters['emission_layers'])
    transition = GatedTransition(z_dim, model_parameters['transmission_dim'])
    encoder = SymplecticODEEncoder(input_dim, z_dim, model_parameters['potential_hidden'],
                                   model_parameters['potential_layers'],
                                   non_linearity='relu', batch_first=True,
                                   rnn_layers=model_parameters['encoder_layers'],
                                   dropout=model_parameters['encoder_dropout_rate'],
                                   order=model_parameters['integrator_order'],
                                   dissipative=True if model_parameters['dissipative'] == 'dissipative' else False,
                                   learn_kinetic=model_parameters['learn_kinetic'],
                                   dt=model_parameters['dt'], discretization=model_parameters['discretization'])

    # create model
    vae = DMM(emitter, transition, combiner, encoder, z_dim,
              (model_parameters['encoder_layers'], model_parameters['batch_size'], z_dim))
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.to(device)
    vae.eval()

    states_mean = checkpoint['state_mu']
    states_std = checkpoint['state_sig']
    obs_mean = checkpoint['obs_mu']
    obs_std = checkpoint['obs_sig']

    # parse system parameters
    m = np.diag(np.array(list(map(float, config['System']['M'].split(',')))))
    n_dof = m.shape[0]
    c = np.reshape(np.array(list(map(float, config['System']['C'].split(',')))), [n_dof, n_dof])
    k = np.reshape(np.array(list(map(float, config['System']['K'].split(',')))), [n_dof, n_dof])
    flow_type = config['System']['Dynamics']

    if flow_type == 'halfcar':
        c2 = np.reshape(np.array(list(map(float, config['System']['C2'].split(',')))), [n_dof, n_dof])
        k2 = np.reshape(np.array(list(map(float, config['System']['K2'].split(',')))), [n_dof, n_dof])
        k3 = np.reshape(np.array(list(map(float, config['System']['K3'].split(',')))), [n_dof, n_dof])

    dissipative = model_parameters['dissipative']

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
        force_fct = lambda x: scipy.signal.unit_impulse(len(tics), [x, x + 1, x + 3])
    elif force_type == 'sinusoidal':
        force_fct = np.sin
    elif force_type == 'road' or force_type == 'sineroad' or force_type == 'traproad':
        pass
    else:
        raise NotImplementedError()

    if flow_type == 'linear':
        vectorfield = linear
    elif flow_type == 'duffing':
        vectorfield = duffing
    elif flow_type == 'halfcar':
        vectorfield = halfcar
        # upsample for halfcar dynamics
        tics = np.linspace(0., t_max, num=int(t_max / dt * 100), endpoint=False)
    elif flow_type == 'pendulum':
        vectorfield, p = pendulum(n_dof)
    else:
        raise NotImplementedError()

    # run simulation
    q0_start = 0.5
    qdot0_start = 0.5
    q0, qdot0 = q0_start * np.ones([n_dof, 1]).squeeze(), qdot0_start * np.ones([n_dof, 1]).squeeze()
    # q0, qdot0 = q0_start * np.array([0.3745, 0.9507]), qdot0_start * np.array([0.7319, 0.5986])
    w0 = np.concatenate([q0, qdot0], axis=0)

    # generate external forces
    force_input = np.zeros([n_dof, len(tics)])
    #exp_amp = force_amp * (0.25 + np.random.random())
    exp_amp = force_amp * 0.3
    exp_freq = force_freq / 3 * 2 #* (0.25 + np.random.random())
    random_number = random.random()
    case = random_number < 0.5
    for dof in force_dof:
        if dof == 0:
            shift = 0
        else:
            shift = force_shift
        if force_type == "impulse":
            impulse_shift = np.random.randint(0, len(tics) // 2)
            force_input[dof, :] = 0.9 * force_fct(impulse_shift)
        elif force_type == "sinusoidal":
            force_input[dof, :] = exp_amp * force_fct(
                2 * np.pi * (exp_freq * tics * dt))
        elif force_type == 'sineroad':
            print("Using sinusoidal road")
            force_input[dof, :] = exp_amp / 2 * scipy.signal.sawtooth(2 * np.pi * exp_freq * tics - shift,
                                                                width=0.5) + exp_amp / 2
        elif force_type == 'traproad':
            print("Using trapezoidal road")
            amp2 = exp_amp * 1.5
            trap_force = amp2 / 2 * scipy.signal.sawtooth(2 * np.pi * exp_freq * tics - shift, width=0.5) + amp2 / 2
            trap_force[trap_force > exp_amp] = exp_amp
            force_input[dof, :] = trap_force
        elif force_type == 'road':
            if case:
                print("Using sinusoidal road")
                force_input[dof, :] = exp_amp / 2 * scipy.signal.sawtooth(2 * np.pi * exp_freq * tics - shift,
                                                                    width=0.5) + exp_amp / 2
            else:
                print("Using trapezoidal road")
                amp2 = exp_amp * 1.5
                trap_force = amp2 / 2 * scipy.signal.sawtooth(2 * np.pi * exp_freq * tics - shift, width=0.5) + amp2 / 2
                trap_force[trap_force > exp_amp] = exp_amp
                force_input[dof, :] = trap_force

    fint = interp1d(tics, force_input, fill_value='extrapolate')

    if flow_type == 'linear':
        p = [m, c, k, fint]
    elif flow_type == 'duffing':
        p = [m, c, k, k / 3, fint]
    elif flow_type == 'halfcar':
        p = [m, c, c2, k, k2, k3, fint]
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

    # subsample state
    if flow_type == 'halfcar':
        state = scipy.signal.decimate(state, 10, axis=0)
        state = scipy.signal.decimate(state, 10, axis=0)

    # calcuate energy of system
    q = state[:, :n_dof, None]
    qdot = state[:, n_dof:2 * n_dof, None]

    if flow_type == "pendulum":
        lenghts = [8, 8]
        potential = -(m[0, 0] + m[1, 1]) * 9.8 * lenghts[0] * np.cos(q[:, 0]) - m[1, 1] * 9.8 * lenghts[1] * np.cos(
            q[:, 1])
        kinetic = 0.5 * m[0, 0] * (lenghts[0] * qdot[:, 0]) ** 2 + \
                  0.5 * m[1, 1] * ((lenghts[0] * qdot[:, 0]) ** 2 + (lenghts[1] * qdot[:, 1]) ** 2
                                   + 2 * lenghts[0] * lenghts[1] * qdot[:, 0] * qdot[:, 1] * np.cos(q[:, 0] - q[:, 1]))
    else:
        kinetic = 0.5 * np.matmul(np.transpose(qdot, axes=[0, 2, 1]), np.matmul(m, qdot))
        potential = 0.5 * np.matmul(np.transpose(q, axes=[0, 2, 1]), np.matmul(k, q))

    obs = np.zeros([state.shape[0], len(obs_idx)])
    for i, idx in enumerate(obs_idx):
        state_signal = scipy.signal.detrend(state[:, idx])
        signal_power = np.mean(state_signal ** 2)
        signal_power_dB = 10 * np.log10(signal_power)
        noise_dB = signal_power_dB - obs_noise[i]
        noise_watt = 10 ** (noise_dB / 10)
        noise = np.random.normal(0, np.sqrt(noise_watt), state.shape[0])
        obs[:, i] = state[:, idx] + noise

    n_len = model_parameters['seq_len'] * 6 - 1
    n_obs = len(obs_idx)
    dt = model_parameters['dt']
    t = np.arange(0, n_len * dt, dt, dtype=float)
    sample_obs = (obs - obs_mean) / obs_std

    k = 1
    sample_obs = torch.from_numpy(sample_obs[:, :n_len + 1, :]).float()
    sample_obs = sample_obs.to(device)
    Z, Z_gen, Z_gen_scale, Obs, Obs_scale = vae.reconstruction(sample_obs, k)
    Obs = Obs.detach() * obs_std + obs_mean
    Obs_scale = Obs_scale.detach() * obs_std + obs_mean
    ground_truth = torch.from_numpy(state[:n_len, obs_idx]).unsqueeze(0).float()
    ground_truth = ground_truth.to(device)
    mse = torch.abs((Obs - ground_truth) / (ground_truth + 1e-6)).mean().item()  # add 1e-6 to avoid inf
    error = torch.logical_or(torch.lt(ground_truth, (Obs - 2 * Obs_scale)),
                             torch.gt(ground_truth, (Obs + 2 * Obs_scale))).float().mean().item()
    t_vec = torch.arange(1, n_len + 1) * dt

    q = Z[:, :, :z_dim // 2]
    qd = Z[:, :, z_dim // 2:]
    qdot = qd
    qdot = qdot[..., None].squeeze(0).detach().numpy()

    if dissipative == 'dissipative':
        input_tensor = torch.cat([t_vec.float().unsqueeze(1), q.squeeze()], dim=1).unsqueeze(0)
    else:
        # input_tensor = torch.cat([torch.from_numpy(q).float(), torch.from_numpy(qd).float()], dim=1)
        input_tensor = q

    latent_potential = vae.encoder.latent_func.energy(t_vec, input_tensor).detach().numpy()

    q = q[..., None].squeeze(0).detach().numpy()
    m = np.eye(z_dim // 2)

    if flow_type == "pendulum":
        latent_kinetic = 0.5 * m[0, 0] * (lenghts[0] * qdot[:, 0]) ** 2 + \
                         0.5 * m[1, 1] * ((lenghts[0] * qdot[:, 0]) ** 2 + (lenghts[1] * qdot[:, 1]) ** 2
                                          + 2 * lenghts[0] * lenghts[1] * qdot[:, 0] * qdot[:, 1] * np.cos(
                    q[:, 0] - q[:, 1]))
    else:
        latent_kinetic = 0.5 * np.matmul(np.transpose(qdot, axes=[0, 2, 1]), np.matmul(m, qdot))

    latent_kinetic = latent_kinetic.flatten()

    ###########################################################################################
    # PLOTS

    import matplotlib.pyplot as plt
    from matplotlib.legend import _get_legend_handles_labels
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    def cm_to_inch(value):
        return value / 2.54

    textwidth = 16.0  # cm
    fontsize = 12  # pts

    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}, size=fontsize)
    plt.rc('text', usetex=True)
    colors = sns.color_palette("tab10", 6)

    # Observation plots
    fig = plt.figure(1, figsize=(16, 8))
    gs = gridspec.GridSpec(n_obs, 1)
    gs.update(wspace=0.1, hspace=1)  # spacing between subplots

    for i, idx in enumerate(obs_idx):
        y1 = state[:n_len, idx]
        y2 = Obs[0, :n_len, i]
        sample_y = obs[:n_len, i]
        bound = Obs_scale[0, :n_len, i]

        xtr_subsplot = fig.add_subplot(gs[i])
        plt.plot(t, y1, linestyle='-', label='ground truth', color=colors[0])  # plot data
        plt.plot(t, y2, linestyle='-', label='predicted mean', color=colors[1])  # plot data
        plt.fill_between(t, y2 - 2 * bound, y2 + 2 * bound, facecolor=colors[2], alpha=0.5,
                         label=r'2-$\sigma$ range')
        plt.scatter(t, sample_y, s=2, color=colors[3], label='noisy samples')
        # plt.legend(loc='upper right')  # add the legend (will default to 'best' location)
        #plt.xlabel('time [s]')

        if flow_type == 'pendulum':
            if idx < n_dof:
                plt.ylabel('angular\ndisplacement\n[rad]')
                plt.title(r"$\theta_{}(t)$".format(idx))
            elif idx < 2 * n_dof:
                plt.ylabel("angular\n" + "velocity\n" + r"[rad/s]")
                plt.title(r"$\dot{{\theta}}_{}(t)$".format(idx - n_dof))
            else:
                plt.ylabel("angular\n" + "acceleration\n" + r"[rad/$s^2$]")
                plt.title(r"$\ddot{{\theta}}_{}(t)$".format(idx - 2 * n_dof))
        else:
            if idx < n_dof:
                plt.ylabel('displacement\n[m]')
                plt.title(r"$z_{}(t)$".format(idx))
            elif idx < 2 * n_dof:
                plt.ylabel("velocity\n" + r"[m/s]")
                plt.title(r"$\dot{{z}}_{}(t)$".format(idx - n_dof))
            else:
                plt.ylabel("acceleration\n" + r"[m/$s^2$]")
                plt.title(r"$\ddot{{z}}_{}(t)$".format(idx - 2 * n_dof))
        plt.xlim([0, (n_len - 1) * dt])
    plt.xlabel('time [s]')
    fig.legend(*_get_legend_handles_labels([fig.axes[0]]))
    fig.set_figwidth(cm_to_inch(textwidth))
    fig.align_ylabels(fig.axes)
    plt.savefig(os.path.join(cfg.root_path, 'experiments', cfg.ckpt_path, ckpt_folder, 'obs.svg'), format='svg', bbox_inches='tight', dpi=1200)
    #plt.show()

    # Phase plot
    latent = Z_gen[0, 5:n_len, :].detach().numpy()
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

    fig = plt.figure(2, figsize=(15, 15))
    plt.plot(ground_mat[:, 0], ground_mat[:, 1], linestyle='dashed', linewidth=1.5, label='true phase')
    plt.scatter(latent_rot[:, 0], latent_rot[:, 1], color='r', label='latent phase (rotated)')
    plt.legend(loc='upper left')
    plt.xlabel(r'$z_0$ (normalized)')
    plt.ylabel(r'$z_1$ (normalized)')

    ratio = 0.7
    fig.set_size_inches(cm_to_inch(textwidth * ratio), cm_to_inch(textwidth * ratio))
    plt.savefig(os.path.join(cfg.root_path, 'experiments', cfg.ckpt_path, ckpt_folder, 'phase.svg'), format='svg', bbox_inches='tight', dpi=1200)
    #plt.show()

    fig = plt.figure(3)
    plt.plot(t[:n_len], latent_kinetic.squeeze()[:n_len], color=colors[0], label='learned kinetic')
    plt.plot(t[:n_len], latent_potential.squeeze()[:n_len], color=colors[1], label='learned potential')
    plt.plot(t[:n_len], kinetic.squeeze()[:n_len], color=colors[-1], label='true kinetic')
    plt.plot(t[:n_len], potential.squeeze()[:n_len], color=colors[-2], label='true potential')
    learned_mechanical = latent_kinetic.squeeze()[:n_len] + latent_potential.squeeze()[:n_len]
    mechanical = kinetic.squeeze()[:n_len] + potential.squeeze()[:n_len]

    plt.plot(t[:n_len], learned_mechanical, color=colors[2], linestyle='--', label='learned mechanical')
    plt.plot(t[:n_len], mechanical, color=colors[-3], linestyle='--', label='true mechanical')

    plt.xlabel(r'time [s]')
    plt.ylabel('Energy [J]')  # label the y axis
    plt.legend(loc='upper right')  # add the legend (will default to 'best' location)
    plt.xlim([0, 10.])
    plt.ylim([0, 1.75])
    fig.set_size_inches(cm_to_inch(textwidth), cm_to_inch(textwidth))
    plt.savefig(os.path.join(cfg.root_path, 'experiments', cfg.ckpt_path, ckpt_folder, 'energy.svg'), format='svg', bbox_inches='tight', dpi=1200)
    #plt.show()

    return 0


if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser(description="parse args")

    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--ckpt-path', type=str, default='3_springmass_linear_free_free')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    test_model(args)
