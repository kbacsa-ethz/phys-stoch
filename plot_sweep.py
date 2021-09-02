from comet_ml import Experiment

import argparse
import os
import json
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_sweep(cfg):

    if cfg.headless:
        mpl.use('Agg')  # if you are on a headless machine
    else:
        mpl.use('TkAgg')

    experiment = Experiment(project_name="2dof", api_key="Bm8mJ7xbMDa77te70th8PNcT8")

    param_sweep = ["emission_dim", "emission_layers", "transmission_dim", "potential_hidden", "potential_layers", "encoder_layers"]

    exp = []
    file_path = os.path.join(cfg.root_path, "sweeps", "2dof.txt")
    with open(file_path) as file:
        while line := file.readline().rstrip():
            out = json.loads(line)
            df = pd.DataFrame.from_dict(out.items())
            df.set_index(0, inplace=True)
            exp.append(df)

    df_final = pd.concat(exp, axis=1)
    df_final = df_final.transpose()

    for param in param_sweep:
        df_final.plot.scatter(param, "mse")
        plt.title("Sweep of {}".format(param))
        experiment.log_figure(figure_name=param)


if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--root-path', type=str, default='.')

    # Machine parameters
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    plot_sweep(args)
