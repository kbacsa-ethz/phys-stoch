import os
import argparse
import configparser
import random

import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rc('font', family='serif')

import pyro

# visualization
from models import *
from dmm import DMM
from utils import data_path_from_config

import scipy.signal as signal
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import scipy.signal

from dynamics import *


def test(cfg):
    # add DLSC parameters like seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    # use gpu
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    config = configparser.ConfigParser()
    config.read(os.path.join(cfg.root_path, cfg.config_path))

    exp_name = data_path_from_config(config)
    obs_idx = list(map(int, config['Simulation']['Observations'].split(',')))

    # parse system parameters
    print(cfg)
    m = np.diag(np.array(list(map(float, config['System']['M'].split(',')))))
    n_dof = m.shape[0]
    c = np.reshape(np.array(list(map(float, config['System']['C'].split(',')))), [n_dof, n_dof])
    k = np.reshape(np.array(list(map(float, config['System']['K'].split(',')))), [n_dof, n_dof])
    flow_type = config['System']['Dynamics']

    if flow_type == 'halfcar':
        c2 = np.reshape(np.array(list(map(float, config['System']['C2'].split(',')))), [n_dof, n_dof])
        k2 = np.reshape(np.array(list(map(float, config['System']['K2'].split(',')))), [n_dof, n_dof])
        k3 = np.reshape(np.array(list(map(float, config['System']['K3'].split(',')))), [n_dof, n_dof])

    if not c.astype(np.int).any():
        dissipative = True
    else:
        dissipative = True

    if flow_type == 'duffing':
        setattr(cfg, 'emission_dim', 14)
        setattr(cfg, 'emission_layers', 1)
        setattr(cfg, 'encoder_layers', 1)
        setattr(cfg, 'potential_hidden', 54)
        setattr(cfg, 'potential_layers', 0)
        setattr(cfg, 'transmission_dim', 36)

        if dissipative:
            """
            states_mean = np.array([[[1.56436474e-02, 9.03638189e-03, 6.61162317e-05, - 8.22759220e-06,
                                      3.11385880e-04, 3.48087424e-04]]])
            states_std = np.array([[[0.34497259, 0.29157891, 0.33275987, 0.32870161, 0.38388944, 0.43193062]]])
            obs_mean = np.array([[[0.01561499, 0.00909796, 0.00018081, - 0.00020756]]])
            obs_std = np.array([[[0.37581299, 0.32815157, 0.41253529, 0.45759588]]])
            """
            states_mean = np.array([[[ 0.01544272,  0.00904093,  0.00064932,  0.00059605, -0.00045951, -0.00035497]]])
            states_std = np.array([[[0.35605751, 0.30015359, 0.34687241, 0.33427918, 0.39633185, 0.43023295]]])
            obs_mean = np.array([[[0.01579435,  0.0089224, -0.00056168, -0.00052515]]])
            obs_std = np.array([[[0.37302853, 0.31474254, 0.39816951, 0.4325108]]])

        elif cfg.partial:
            states_mean = np.array([[[0.00638681, 0.00610086, -0.00869065, -0.00836581, -0.00754972,
                                      -0.00812269]]])
            states_std = np.array([[[0.56874353, 0.50638157, 0.71662655, 0.68666006, 1.15918821, 1.24995868]]])
            obs_mean = np.array([[[0.00562431, -0.00751712]]])
            obs_std = np.array([[[0.59655451, 1.16493323]]])
        else:
            states_mean = np.array([[[-0.00018105, - 0.00066574, 0.00030546, 0.00049744, 0.00026192, 0.00142136]]])
            states_std = np.array([[[0.92813288, 0.92993491, 1.90475785, 2.15138085, 5.02098828, 5.97846263]]])
            obs_mean = np.array([[[-0.00066353, - 0.0009563, 0.00039857, 0.00119788]]])
            obs_std = np.array([[[0.94974715, 0.95087484, 5.02525429, 5.98210665]]])
    elif flow_type == 'linear':
        if n_dof == 2:
            setattr(cfg, 'emission_dim', 18)
            setattr(cfg, 'emission_layers', 2)
            setattr(cfg, 'encoder_layers', 1)
            setattr(cfg, 'potential_hidden', 85)
            setattr(cfg, 'potential_layers', 2)
            setattr(cfg, 'transmission_dim', 15)
            if dissipative:
                states_mean = np.array([[[-0.00052196, - 0.00056236, 0.00019102, 0.00066107, 0.00031078,
                                          0.00087969]]])
                states_std = np.array([[[0.58341699, 0.55143364, 0.83937896, 0.90896708, 1.56943701, 1.83393094]]])
                obs_mean = np.array([[[-0.00100444, - 0.00085291, 0.00044744, 0.0006562]]])
                obs_std = np.array([[[0.6161545, 0.58673665, 1.5817723, 1.84504716]]])
            elif cfg.partial:
                states_mean = np.array([[[-0.00363647, -0.00211204, -0.00809523, -0.00677396, 0.00562376,
                                          0.00018717]]])
                states_std = np.array([[[0.60315667, 0.517374, 0.68331947, 0.63447033, 0.95472971, 1.01317489]]])
                obs_mean = np.array([[[-0.00445266, 0.00565991]]])
                obs_std = np.array([[[0.63256084, 0.95954157]]])
            else:
                states_mean = np.array([[[0.00018988, - 0.00016362, 0.00051457, 0.00066631, - 0.00081405, 0.00091053]]])
                states_std = np.array([[[1.022071, 0.99691651, 1.57848652, 1.74890221, 3.08154555, 3.63055205]]])
                obs_mean = np.array([[[-0.0002926, - 0.00045418, - 0.0006774, 0.00068704]]])
                obs_std = np.array([[[1.04057759, 1.017169, 3.08788495, 3.63614872]]])
        elif n_dof == 3:
            setattr(cfg, 'emission_dim', 29)
            setattr(cfg, 'emission_layers', 1)
            setattr(cfg, 'encoder_layers', 1)
            setattr(cfg, 'potential_hidden', 11)
            setattr(cfg, 'potential_layers', 4)
            setattr(cfg, 'transmission_dim', 24)
            if cfg.partial:
                states_mean = np.array([[[3.01244370e-02, 3.20745877e-02, 2.39366314e-02, 1.66729464e-03,
                                          1.55803643e-05, -2.85578676e-03, -1.88880414e-02, -1.55867027e-02,
                                          -1.68425414e-02]]])
                states_std = np.array([[[0.66622605, 0.71405323, 0.53527093, 0.60439193, 0.65792662, 0.53235315,
                                         0.84385351, 1.01646212, 0.77683493]]])
                obs_mean = np.array([[[0.03064304, 0.02985496]]])
                obs_std = np.array([[[0.92663692, 1.10736226]]])
            else:
                states_mean = np.array([[[0.02992056, 0.03181143, 0.02422422, 0.0021101, 0.00104457,
                                          -0.00277293, -0.01884777, -0.01466999, -0.01799172]]])
                states_std = np.array([[[0.66594007, 0.71419109, 0.53601297, 0.6089117, 0.6664796, 0.53728351,
                                         0.86351435, 1.04844165, 0.7927688]]])
                obs_mean = np.array([[[0.02994358, 0.03095564, 0.02378642, -0.01875875, -0.0156254,
                                       -0.01807897]]])
                obs_std = np.array([[[0.7301821, 0.77472186, 0.61419106, 0.9136632, 1.09087811, 0.84776676]]])
        else:
            raise NotImplementedError
    elif flow_type == 'pendulum':
        setattr(cfg, 'emission_dim', 13)
        setattr(cfg, 'emission_layers', 0)
        setattr(cfg, 'encoder_layers', 1)
        setattr(cfg, 'potential_hidden', 21)
        setattr(cfg, 'potential_layers', 4)
        setattr(cfg, 'transmission_dim', 18)
        if cfg.partial:
            states_mean = np.array([[[0.00604129, -0.01429281, -0.01235269, -0.01564474, -0.00581241,
                                      -0.00448398]]])
            states_std = np.array([[[0.53305792, 1.30761524, 0.52879144, 0.79133918, 0.82188762, 1.0926236]]])
            obs_mean = np.array([[[0.00529704, -0.00575641]]])
            obs_std = np.array([[[0.55898133, 0.82614014]]])
        else:
            states_mean = np.array(
                [[[0.00471831, - 0.02527156, - 0.01187413, - 0.01469576, - 0.00447611, -0.00421897]]])
            states_std = np.array([[[0.51161866, 1.60491627, 0.51462374, 0.77564956, 0.80426609, 1.07328515]]])
            obs_mean = np.array([[[0.00423583, - 0.02556212, - 0.00433946, - 0.00444246]]])
            obs_std = np.array([[[0.54962711, 1.61728878, 0.8287365, 1.092226]]])
    elif flow_type == 'halfcar':
        states_mean = np.array([[[2.69076545e-05, 4.20947715e-06, 8.49376468e-06, 4.53657810e-05, 1.23947558e-07,
                                  1.53617007e-05, 1.72890732e-06, 2.54777796e-07, 5.70140182e-07, 2.96521391e-06,
                                  -2.31731249e-08, 1.01003874e-06, -7.78982798e-06, -5.12575901e-08, -1.81307585e-09,
                                  -4.50203552e-06, 7.85709930e-09, 1.11824855e-08]]])
        states_std = np.array([[[1.16669297e-05, 3.58419126e-06, 4.20733563e-06, 1.98268094e-05, 4.69122337e-06,
                                 7.97139032e-06, 6.92979995e-05, 2.03063194e-05, 5.29559543e-06, 1.10978577e-04,
                                 3.27961276e-05, 3.44373859e-05, 2.62811825e-03, 1.57434850e-04, 3.78882188e-05,
                                 3.05877178e-03, 3.05423202e-04, 3.13693015e-04]]])
        obs_mean = np.array([[[2.69092946e-05, 4.21022732e-06, 8.49377738e-06, 4.53667214e-05, 1.21823658e-07,
                               1.53609285e-05, -7.79127042e-06, -4.98090840e-08, -2.28254871e-09, -4.50198628e-06,
                               9.56905701e-09, 1.00218802e-08]]])
        obs_std = np.array([[[1.18371004e-05, 3.72060126e-06, 4.23712534e-06, 1.99806990e-05, 5.09556472e-06,
                              8.21861681e-06, 2.62811652e-03, 1.57445926e-04, 3.78910385e-05, 3.05877496e-03,
                              3.05427738e-04, 3.13701496e-04]]])
    else:
        raise NotImplemented

    # Save normalization parameters
    print("states_mean: {}".format(states_mean))
    print("states_std: {}".format(states_std))
    print("obs_mean: {}".format(obs_mean))
    print("obs_std: {}".format(obs_std))

    if dissipative:
        checkpoint = torch.load("checkpoints/{}_{}_dissipative_with.pth".format(flow_type, n_dof))
    elif cfg.partial:
        checkpoint = torch.load("checkpoints/{}_{}_partial.pth".format(flow_type, n_dof))
    else:
        print('{}_{}.pth'.format(flow_type, n_dof))
        checkpoint = torch.load("checkpoints/{}_{}.pth".format(flow_type, n_dof))

    # modules
    input_dim = len(obs_idx)

    if cfg.partial:
        z_dim = int(states_mean.shape[-1] * 2 / 3)
    else:
        z_dim = int(states_mean.shape[-1] * 2 / 3)

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
    vae.eval()

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
    q0_start = 1e-6
    qdot0_start = 1e-6
    q0, qdot0 = q0_start * np.ones([n_dof, 1]).squeeze(), qdot0_start * np.ones([n_dof, 1]).squeeze()
    # q0, qdot0 = q0_start * np.array([0.3745, 0.9507]), qdot0_start * np.array([0.7319, 0.5986])
    w0 = np.concatenate([q0, qdot0], axis=0)

    # generate external forces
    force_input = np.zeros([n_dof, len(tics)])
    #exp_amp = force_amp * (0.25 + np.random.random())
    exp_amp = force_amp * 0.3
    exp_freq = force_freq / 3 * 2 #* (0.25 + np.random.random())
    random_number = random.random()
    print(random_number)
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
            force_input[dof, :] = exp_amp / 2 * signal.sawtooth(2 * np.pi * exp_freq * tics - shift,
                                                                width=0.5) + exp_amp / 2
        elif force_type == 'traproad':
            print("Using trapezoidal road")
            amp2 = exp_amp * 1.5
            trap_force = amp2 / 2 * signal.sawtooth(2 * np.pi * exp_freq * tics - shift, width=0.5) + amp2 / 2
            trap_force[trap_force > exp_amp] = exp_amp
            force_input[dof, :] = trap_force
        elif force_type == 'road':
            if case:
                print("Using sinusoidal road")
                force_input[dof, :] = exp_amp / 2 * signal.sawtooth(2 * np.pi * exp_freq * tics - shift,
                                                                    width=0.5) + exp_amp / 2
            else:
                print("Using trapezoidal road")
                amp2 = exp_amp * 1.5
                trap_force = amp2 / 2 * signal.sawtooth(2 * np.pi * exp_freq * tics - shift, width=0.5) + amp2 / 2
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
        state = signal.decimate(state, 10, axis=0)
        state = signal.decimate(state, 10, axis=0)

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

    n_len = cfg.seq_len * 12 - 1
    n_obs = len(obs_idx)
    dt = cfg.dt
    time = np.arange(0, n_len * dt, dt, dtype=float)
    obs = (obs - obs_mean) / obs_std

    k = 1
    sample_obs = torch.from_numpy(obs[:, :n_len + 1, :]).float()
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

    if cfg.dissipative:
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

    print(mse)
    print(error)
    colors = sns.color_palette("viridis", 4)

    # normalize energy
    latent_potential = latent_potential[:, 5:, :]
    latent_kinetic = latent_kinetic[5:]

    latent_kinetic = (latent_kinetic - latent_kinetic.min()) / (latent_kinetic.max() - latent_kinetic.min())
    latent_potential = (latent_potential - latent_potential.min()) / (latent_potential.max() - latent_potential.min())
    kinetic = kinetic[:n_len]
    potential = potential[:n_len]

    kinetic = (kinetic - kinetic.min()) / (kinetic.max() - kinetic.min())
    potential = (potential - potential.min()) / (potential.max() - potential.min())

    fig = plt.figure(1, figsize=(20, 10))
    gs = gridspec.GridSpec(n_obs // 2, n_obs // (n_obs // 2))
    gs.update(wspace=0.2, hspace=0.25)  # spacing between subplots

    list_of_names = [r'$\theta_{}$'.format(i + 1) for i in range(n_dof)] + [r'$\ddot{{\theta}}_{}$'.format(i + 1) for i
                                                                            in range(n_dof)]
    learned_mechanical = latent_kinetic.squeeze() + latent_potential.squeeze()
    mechanical = kinetic.squeeze() + potential.squeeze()

    # save Obs, state, sample_obs, obs_std, obs_mean, Obs_scale
    np.save('intermediate/Obs.npy', Obs.detach().numpy())
    np.save('intermediate/Z_gen.npy', Z.detach().numpy())
    np.save('intermediate/state.npy', state)
    np.save('intermediate/sample_obs.npy', sample_obs.detach().numpy() * obs_std + obs_mean)
    np.save('intermediate/Obs_scale.npy', Obs_scale.detach().numpy())

    np.save('intermediate/latent_kinetic.npy', latent_kinetic)
    np.save('intermediate/latent_potential.npy', latent_potential)
    np.save('intermediate/kinetic.npy', kinetic)
    np.save('intermediate/potential.npy', potential)
    np.save('intermediate/learned_mechanical.npy', learned_mechanical)
    np.save('intermediate/mechanical.npy', mechanical)

    """

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
        plt.ylabel(list_of_names[i], fontsize=14)  # label the y axis

        plt.legend(loc='upper right')  # add the legend (will default to 'best' location)

    plt.suptitle("k+{}-predictions".format(k))
    plt.show()

    # plot phase
    fig = plt.figure(2, figsize=(20, 10))
    gs = gridspec.GridSpec(z_dim // 2, z_dim // 2)

    # normalize and remove starting points
    latent = Z[:, 5:n_len, :].squeeze().detach().numpy()
    ground = state[5:n_len, :]
    latent = (latent - latent.min()) / (latent.max() - latent.min() + 1e-6)
    ground = (ground - ground.min()) / (ground.max() - ground.min() + 1e-6)

    # regular phase
    for i in range(z_dim // 2):
        xtr_subsplot = fig.add_subplot(gs[i])

        x, y = latent[:, i], latent[:, i + z_dim // 2]
        x_g, y_g = ground[:, obs_idx[i]], ground[:, obs_idx[i + z_dim // 2]]

        latent_mat = np.stack([x, y], axis=0)
        ground_mat = np.stack([x_g, y_g], axis=0)
        T, _, _, _ = np.linalg.lstsq(latent_mat, ground_mat, rcond=None)
        latent_rot = latent_mat @ T

        plt.xlabel(r'$\theta_1$ (normalized)', fontsize=11)
        plt.ylabel(r'$\dot{\theta}_1$ (normalized)', fontsize=11)
        plt.plot(x_g, y_g, color=colors[0], linestyle='dashed', linewidth=0.5, label='original')
        plt.scatter(latent_rot[0, :], latent_rot[1, :], color=colors[-1], label='predicted (rotated)')
        plt.legend(fontsize=11, loc='upper right')  # add the legend (will default to 'best' location)

    
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
    max_iter = 150
    plt.plot(time[:max_iter], latent_kinetic.squeeze()[:max_iter], color=colors[0], label='learned kinetic')
    plt.plot(time[:max_iter], latent_potential.squeeze()[:max_iter], color=colors[1], label='learned potential')
    plt.plot(time[:max_iter], kinetic.squeeze()[:max_iter], color=colors[-1], label='true kinetic')
    plt.plot(time[:max_iter], potential.squeeze()[:max_iter], color=colors[-2], label='true potential')
    learned_mechanical = latent_kinetic.squeeze()[:max_iter] + latent_potential.squeeze()[:max_iter]
    mechanical = kinetic.squeeze()[:max_iter] + potential.squeeze()[:max_iter]

    plt.plot(time[:max_iter], learned_mechanical, color=colors[2], linestyle='--', label='learned mechanical')
    plt.plot(time[:max_iter], mechanical, color=colors[-3], linestyle='--', label='true mechanical')

    plt.xlabel(r'time [s]', fontsize=14)
    plt.ylabel('Energy', fontsize=14)  # label the y axis
    plt.legend(fontsize=11, loc='upper right')  # add the legend (will default to 'best' location)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.75])
    plt.show()
    """
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--config-path', type=str, default='config/2springmass_duffing_sinusoidal_dissipative.ini')
    parser.add_argument('-e', '--emission-dim', type=int, default=18)
    parser.add_argument('-ne', '--emission-layers', type=int, default=2)
    parser.add_argument('-tr', '--transmission-dim', type=int, default=15)
    parser.add_argument('-ph', '--potential-hidden', type=int, default=85)
    parser.add_argument('-pl', '--potential-layers', type=int, default=2)
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
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--comet', action='store_true')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    test(args)
