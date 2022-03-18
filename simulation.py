import argparse
import configparser
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import scipy.signal as signal
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from dynamics import *
from utils import data_path_from_config, phase_shift


def main(ag, cfg):
    # parse system parameters
    system_type = cfg['System']['Name']
    flow_type = cfg['System']['Dynamics']
    m = np.diag(np.array(list(map(float, cfg['System']['M'].split(',')))))
    n_dof = m.shape[0]
    c = np.reshape(np.array(list(map(float, cfg['System']['C'].split(',')))), [n_dof, n_dof])
    k = np.reshape(np.array(list(map(float, cfg['System']['K'].split(',')))), [n_dof, n_dof])

    # parse external forces
    # TODO Add additional types of forces
    force_type = cfg['Forces']['Type']
    force_amp = float(cfg['Forces']['Amplitude'])
    force_freq = float(cfg['Forces']['Frequency'])
    force_shift = float(cfg['Forces']['Shift'])
    force_dof = np.array(list(map(int, cfg['Forces']['Inputs'].split(','))))

    # parse simulation parameters
    seed = int(cfg['Simulation']['Seed'])
    n_iter = int(cfg['Simulation']['Iterations'])
    obs_idx = np.array(list(map(int, cfg['Simulation']['Observations'].split(','))))
    obs_noise = np.array(list(map(float, cfg['Simulation']['Noise'].split(','))))
    x_min = float(cfg['Simulation']['Lower_x'])
    x_max = float(cfg['Simulation']['Upper_x'])
    y_min = float(cfg['Simulation']['Lower_xdot'])
    y_max = float(cfg['Simulation']['Upper_xdot'])
    abserr = float(cfg['Simulation']['Absolute'])
    relerr = float(cfg['Simulation']['Relative'])
    dt = float(cfg['Simulation']['Delta'])
    t_max = float(cfg['Simulation']['Time'])

    # fix random seed for reproducibility
    np.random.seed(seed)

    experiment_tree = data_path_from_config(cfg)
    save_path = os.path.join(ag.root_path, 'data', system_type)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    tics = np.linspace(0., t_max, num=int(t_max / dt), endpoint=False)

    if force_type == 'free':
        force_fct = lambda x: 0
    elif force_type == 'impulse':
        force_fct = lambda x: signal.unit_impulse(len(tics), [x, x+1, x+3])
    elif force_type == 'sinusoidal':
        force_fct = np.sin
    else:
        raise NotImplementedError()

    state_tensor = np.zeros([n_iter, len(tics), 3 * n_dof])
    obs_tensor = np.zeros([n_iter, len(tics), len(obs_idx)])
    force_tensor = np.zeros([n_iter, len(tics), n_dof])

    # kinetic, potential, dissipative, input, total
    energy_tensor = np.zeros([n_iter, len(tics), 5])

    if flow_type == 'linear':
        vectorfield = linear
    elif flow_type == 'duffing':
        vectorfield = duffing
    elif flow_type == 'pendulum':
        vectorfield, p = pendulum(n_dof)
    else:
        raise NotImplementedError()

    # run simulation
    for iter_idx in tqdm(range(n_iter)):
        # initialize state
        q0 = (x_max - x_min) * np.random.random([n_dof, 1]).squeeze() + x_min
        qdot0 = (y_max - y_min) * np.random.random([n_dof, 1]).squeeze() + y_min
        w0 = np.concatenate([q0, qdot0], axis=0)

        # generate external forces
        force_input = np.zeros([n_dof, len(tics)])
        for dof in force_dof:
            if force_type == "impulse":
                impulse_shift = np.random.randint(0, len(tics) // 2)
                force_input[dof, :] = (force_amp * (0.25 + np.random.random())) * force_fct(impulse_shift)
            if force_type == "sinusoidal":
                force_input[dof, :] = (force_amp * np.random.random()) * force_fct(2*np.pi*(force_freq*np.random.random())*tics*dt)

        fint = interp1d(tics, force_input, fill_value='extrapolate')

        if flow_type == 'linear':
            p = [m, c, k, fint]
        if flow_type == 'duffing':
            p = [m, c, k, k/3, fint]
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
        qdot = state[:, n_dof:2*n_dof, None]
        kinetic = 0.5 * np.matmul(np.transpose(qdot, axes=[0, 2, 1]), np.matmul(m, qdot))
        potential = 0.5 * np.matmul(np.transpose(q, axes=[0, 2, 1]), np.matmul(k, q))

        obs = np.zeros([state.shape[0], len(obs_idx)])
        for i, idx in enumerate(obs_idx):
            obs[:, i] = state[:, idx] + np.random.randn(state.shape[0]) * obs_noise[i]

        state_tensor[iter_idx] = state
        obs_tensor[iter_idx] = obs
        force_tensor[iter_idx] = force_input.T
        energy_tensor[iter_idx, :, 0] = kinetic.flatten()
        energy_tensor[iter_idx, :, 1] = potential.flatten()
        energy_tensor[iter_idx, :, 4] = kinetic.flatten() + potential.flatten()

    np.save(os.path.join(save_path, 'state.npy'), state_tensor, allow_pickle=True)
    np.save(os.path.join(save_path, 'state_param.npy'), np.concatenate([state_tensor.mean(axis=(0, 1)), state_tensor.std(axis=(0, 1))]))
    np.save(os.path.join(save_path, 'obs.npy'), obs_tensor, allow_pickle=True)
    np.save(os.path.join(save_path, 'obs_param.npy'), np.concatenate([obs_tensor.mean(axis=(0, 1)), obs_tensor.std(axis=(0, 1))]))
    np.save(os.path.join(save_path, 'force.npy'), force_tensor, allow_pickle=True)
    np.save(os.path.join(save_path, 'energy.npy'), energy_tensor, allow_pickle=True)

    return 0


if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--config-path', type=str, default='config/2springmass_duffing_free_free.ini')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.root_path, args.config_path))
    main(args, config)
