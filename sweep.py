import argparse
import os
from pathlib import Path
import numpy as np
from train import train


def sweep(cfg):
    # Only look at early steps
    setattr(cfg, "num_epochs", 3)

    Path(os.path.join(cfg.root_path, "sweeps")).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(cfg.root_path, "sweeps", "{}.txt".format(cfg.parameter)), "w") as filep:
        for param_value in range(cfg.sweep_min, cfg.sweep_max, cfg.sweep_step):
            setattr(cfg, cfg.parameter, param_value)
            val_loss = train(cfg)
            aic = 2 * param_value - 2 * np.log(val_loss)
            print("AIC for {} of value {} is {}".format(cfg.parameter, param_value, aic))
            filep.write("{},{:.2f}\n".format(param_value, aic))

    # loop at get AIC
    return 0


if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser(description="parse args")

    # I/O parameters
    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--config-path', type=str, default='config/2springmass_free.ini')

    # Network parameters
    parser.add_argument('-e', '--emission-dim', type=int, default=16)
    parser.add_argument('-ne', '--emission-layers', type=int, default=0)
    parser.add_argument('-tr', '--transmission-dim', type=int, default=32)
    parser.add_argument('-ph', '--potential-hidden', type=int, default=60)
    parser.add_argument('-pl', '--potential-layers', type=int, default=2)
    parser.add_argument('-nenc', '--encoder-layers', type=int, default=2)
    parser.add_argument('-symp', '--symplectic-integrator', type=str, default='velocity_verlet')
    parser.add_argument('--dissipative', action='store_true')
    parser.add_argument('-dt', '--dt', type=float, default=0.1)
    parser.add_argument('-disc', '--discretization', type=int, default=3)

    # Training parameters
    parser.add_argument('-n', '--num-epochs', type=int, default=10)
    parser.add_argument('-te', '--tuning-epochs', type=int, default=10)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('-b1', '--beta1', type=float, default=0.96)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip-norm', type=float, default=10.0)
    parser.add_argument('-lrd', '--lr-decay', type=float, default=0.99996)
    parser.add_argument('-wd', '--weight-decay', type=float, default=2.0)
    parser.add_argument('-bs', '--batch-size', type=int, default=256)
    parser.add_argument('-sq', '--seq-len', type=int, default=50)
    parser.add_argument('-ae', '--annealing-epochs', type=int, default=2)
    parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.5)
    parser.add_argument('-rdr', '--encoder-dropout-rate', type=float, default=0.1)
    parser.add_argument('-iafs', '--num-iafs', type=int, default=0)
    parser.add_argument('-id', '--iaf-dim', type=int, default=100)
    parser.add_argument('-vf', '--validation-freq', type=int, default=1)
    parser.add_argument('--learn-kinetic', action='store_true')

    # Sweep parameters
    parser.add_argument('--parameter', type=str, default='emission_dim')
    parser.add_argument('--sweep-min', type=int, default=8)
    parser.add_argument('--sweep-max', type=int, default=80)
    parser.add_argument('--sweep-step', type=int, default=8)

    # Machine parameters
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--comet', action='store_true')
    parser.add_argument('--headless', action='store_true')

    args = parser.parse_args()
    sweep(args)
