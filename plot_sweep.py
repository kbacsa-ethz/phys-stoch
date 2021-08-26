from comet_ml import Experiment

import argparse
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_sweep(cfg):

    if cfg.headless:
        mpl.use('Agg')  # if you are on a headless machine
    else:
        mpl.use('TkAgg')

    experiment = Experiment(project_name="phys-sweep", api_key="Bm8mJ7xbMDa77te70th8PNcT8")

    data = pd.read_csv(os.path.join(cfg.root_path, "sweeps", cfg.parameter + '.txt'), sep=',', header=None)
    data = pd.DataFrame(data)

    x = data[0]
    y = data[1]
    z = data[2]

    fig = plt.figure(figsize=(16, 7))
    plt.title(cfg.parameter)
    plt.plot(x, y, 'r')
    plt.xlabel("parameter value")
    plt.ylabel("AIC")
    experiment.log_figure(figure=fig, figure_name=cfg.parameter)

    fig = plt.figure(figsize=(16, 7))
    plt.title(cfg.parameter)
    plt.plot(x, z, 'r')
    plt.xlabel("parameter value")
    plt.ylabel("MSE")
    experiment.log_figure(figure=fig, figure_name=cfg.parameter)
    return 0


if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser(description="parse args")

    # I/O parameters
    parser.add_argument('--root-path', type=str, default='.')

    # Sweep parameters
    parser.add_argument('--parameter', type=str, default='emission_dim')

    # Machine parameters
    parser.add_argument('--headless', action='store_true')

    args = parser.parse_args()
    plot_sweep(args)