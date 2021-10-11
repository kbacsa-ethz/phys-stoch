import argparse
import configparser
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from scipy.integrate import odeint
from scipy.interpolate import interp1d

from dynamics import *
from utils import data_path_from_config


def ftle(dphi, T):
    delta = dphi.T @ dphi

    if len(dphi) > 5:
        ftle_val = 1 / np.abs(T) * np.sqrt(max(np.linalg.eigvals(delta)))
    else:
        ftle_val = 1 / np.abs(T) * np.sqrt(delta)
    return ftle_val


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
    force_sig = float(cfg['Forces']['Shift'])
    force_dof = np.array(list(map(int, cfg['Forces']['Inputs'].split(','))))

    # parse simulation parameters
    seed = int(cfg['Simulation']['Seed'])
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

    tics = np.linspace(0., t_max, num=int(t_max / dt), endpoint=False)

    if force_type == 'free':
        force_fct = lambda x: 0
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

    # run grid simulation
    nx, ny = (10, 10)
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    xv, yv = np.meshgrid(x, y)

    ftle_mat = np.zeros([nx, ny])

    for i in range(nx):
        for j in range(ny):
            w0 = np.array([x[i], x[i], y[j], y[j]])
            # Call the ODE solver.

            # generate external forces
            force_input = np.zeros([n_dof, len(tics)])
            for dof in force_dof:
                force_input[dof, :] = (force_amp * np.random.random()) * force_fct(
                    2 * np.pi * force_freq * tics + np.random.random() * force_sig)

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

            ftle_mat[i, j] = ftle(wsol[-1], t_max)

    plt.imshow(ftle_mat, extent=[x_min, x_max, y_min, y_max])
    plt.show()

    return 0


if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--config-path', type=str, default='config/2springmass_duffing_free.ini')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.root_path, args.config_path))
    main(args, config)
