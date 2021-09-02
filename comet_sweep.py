import comet_ml

import argparse
import configparser
import os
import json
import numpy as np
import matplotlib as mpl
import torch
import pyro
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    config_enumerate,
)
from pyro.optim import ClippedAdam

from phys_data import TrajectoryDataset
from utils import data_path_from_config
from models import Emitter, GatedTransition, Combiner, SymplecticODEEncoder
from dmm import DMM


def build_model_graph(experiment):
    # modules
    emitter = Emitter(
        experiment.get_parameter("input_dim"),
        experiment.get_parameter("z_dim"),
        experiment.get_parameter("emission_dim"),
        experiment.get_parameter("emission_layers")
    )
    transition = GatedTransition(
        experiment.get_parameter("z_dim"),
        experiment.get_parameter("transmission_dim")
    )
    combiner = Combiner(
        experiment.get_parameter("z_dim"),
        experiment.get_parameter("z_dim")
    )
    encoder = SymplecticODEEncoder(
        experiment.get_parameter("input_dim"),
        experiment.get_parameter("z_dim"),
        experiment.get_parameter("potential_hidden"),
        experiment.get_parameter("potential_layers"),
        non_linearity='relu',
        batch_first=True,
        rnn_layers=experiment.get_parameter("encoder_layers"),
        dropout=experiment.get_parameter("encoder_dropout_rate"),
        integrator="velocity_verlet",
        dissipative=experiment.get_parameter("dissipative"),
        learn_kinetic=experiment.get_parameter("learn_kinetic"),
        dt=experiment.get_parameter("dt"),
        discretization=experiment.get_parameter("discretization")
    )

    # create model
    vae = DMM(emitter, transition, combiner, encoder, experiment.get_parameter("z_dim"),
              (experiment.get_parameter("encoder_layers"), experiment.get_parameter("batch_size"),
               experiment.get_parameter("z_dim")))
    vae.to(device)
    return vae


def train(experiment, vae, dataloader):
    # setup optimizer
    adam_params = {"lr": experiment.get_parameter("learning_rate"),
                   "betas": (experiment.get_parameter("beta1"), experiment.get_parameter("beta2")),
                   "clip_norm": experiment.get_parameter("clip_norm"), "lrd": experiment.get_parameter("lr_decay"),
                   "weight_decay": experiment.get_parameter("weight_decay")}
    adam = ClippedAdam(adam_params)
    elbo = Trace_ELBO()
    svi = SVI(vae.model, vae.guide, adam, loss=elbo)
    global_step = 0
    annealing_epochs = experiment.get_parameter("annealing_epochs")
    batch_size = experiment.get_parameter("batch_size")

    for epoch in range(experiment.get_parameter("epochs")):
        epoch_loss = 0
        for which_batch, sample in enumerate(dataloader):
            if annealing_epochs > 0 and epoch < annealing_epochs:
                # compute the KL annealing factor approriate for the current mini-batch in the current epoch
                min_af = experiment.get_parameter("minimum_annealing_factor")
                annealing_factor = min_af + (1.0 - min_af) * \
                                   (float(which_batch + epoch * len(dataloader) // batch_size + 1) /
                                    float(cfg.annealing_epochs * len(dataloader) // batch_size))
            else:
                # by default the KL annealing factor is unity
                annealing_factor = 1.0

            mini_batch = sample['obs'].float().to(device)
            mini_batch_mask = torch.ones(
                [mini_batch.size(0), mini_batch.size(1)]).to(
                device)  # assumption that all sequences have the same length

            # do an actual gradient step
            loss = svi.step(mini_batch, mini_batch_mask, annealing_factor)
            epoch_loss += loss

            # record loss
            global_step += 1
            batch_loss = loss / batch_size
            experiment.log_metric("training_loss", batch_loss, step=global_step)
            optim_state = svi.optim.get_state()
            batch_lr = optim_state[next(iter(optim_state))]['param_groups'][0]['lr']
            experiment.log_metric("learning_rate", batch_lr, step=global_step)

        epoch_loss /= len(dataloader)
        print("Mean training loss at epoch {} is {}".format(epoch, epoch_loss))

    return svi


def evaluate(experiment, vae, svi, val_loader, states_normalize, observations_normalize):
    vae.encoder.eval()
    val_epoch_loss = 0
    for sample in val_loader:
        mini_batch = sample['obs'].float().to(device)
        mini_batch_mask = torch.ones([mini_batch.size(0), mini_batch.size(1)]).to(device)

        # do an actual gradient step
        val_epoch_loss += svi.evaluate_loss(mini_batch, mini_batch_mask)

    # record loss and save
    val_epoch_loss /= len(val_loader)
    experiment.log_metric("validation_loss", val_epoch_loss)
    print("Mean validation loss {}:".format(val_epoch_loss))

    # Zhilu plot
    n_re = 0
    n_len = cfg.seq_len * 10
    sample = np.expand_dims(observations_normalize[n_re], axis=0)
    sample = torch.from_numpy(sample[:, : n_len + 1, :]).float()
    sample = sample.to(device)
    Z, Z_gen, Z_gen_scale, Obs, Obs_scale = vae.reconstruction(sample)

    ground_truth = np.expand_dims(states_normalize[n_re], axis=0)
    ground_truth = torch.from_numpy(ground_truth[:, : n_len, obs_idx]).float()
    ground_truth = ground_truth.to(device)
    mse_loss = ((ground_truth - Obs) ** 2).mean().item()
    experiment.log_metric("mse_loss", mse_loss)
    print("Mean error is {}".format(mse_loss))
    return val_epoch_loss, mse_loss


def get_dataset(cfg):
    config = configparser.ConfigParser()
    config.read(os.path.join(cfg.root_path, cfg.config_path))
    exp_name = data_path_from_config(config)
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

    n_exp = states.shape[0]
    observations_windowed = []
    states_windowed = []
    forces_windowed = []
    for t in range(observations.shape[1] - cfg.seq_len):
        qqd_w = observations_normalize[:, t:t + cfg.seq_len + 1]
        observations_windowed.append(qqd_w[:, None])
        qqd_w = states_normalize[:, t:t + cfg.seq_len + 1]
        states_windowed.append(qqd_w[:, None])
        # TODO Normalize forces ?
        qqd_w = forces[:, t:t + cfg.seq_len + 1]
        forces_windowed.append(qqd_w[:, None])

    observations_windowed = np.concatenate(observations_windowed, axis=1)
    states_windowed = np.concatenate(states_windowed, axis=1)
    forces_windowed = np.concatenate(forces_windowed, axis=1)

    # 80/10/10 split by default
    indexes = np.arange(n_exp)
    train_idx = indexes[:int(0.8 * len(indexes))]
    test_idx = indexes[int(0.8 * len(indexes)):]

    train_dataset = TrajectoryDataset(states_windowed[train_idx], observations_windowed[train_idx],
                                      forces_windowed[train_idx], obs_idx,
                                      cfg.seq_len + 1)
    test_dataset = TrajectoryDataset(states_windowed[test_idx], observations_windowed[test_idx],
                                     forces_windowed[test_idx],
                                     obs_idx, cfg.seq_len + 1)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)

    return train_loader, test_loader, observations_normalize, states_normalize


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

    # Machine parameters
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--comet', action='store_true')
    parser.add_argument('--headless', action='store_true')

    cfg = parser.parse_args()

    # add DLSC parameters like seed
    seed = 42
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    if cfg.headless:
        mpl.use('Agg')  # if you are on a headless machine
    else:
        mpl.use('TkAgg')

    # use gpu
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    run_dir = os.path.join(cfg.root_path, 'experiments')

    config = configparser.ConfigParser()
    config.read(os.path.join(cfg.root_path, cfg.config_path))

    obs_idx = list(map(int, config['Simulation']['Observations'].split(',')))

    # Get the dataset:
    loader_train, loader_test, observations_normalize, states_normalize = get_dataset(cfg)

    # The optimization config:
    config = {
        "algorithm": "bayes",
        "name": "Optimize DMM",
        "spec": {"maxCombo": 10, "objective": "minimize", "metric": "loss", "seed": seed},
        "parameters": {
            # sweep parameters
            "emission_dim": {"type": "integer", "scalingType": "linear", "min": 10, "max": 20},
            "emission_layers": {"type": "integer", "scalingType": "linear", "min": 0, "max": 3},
            "transmission_dim": {"type": "integer", "scalingType": "linear", "min": 20, "max": 40},
            "potential_hidden": {"type": "integer", "scalingType": "linear", "min": 10, "max": 100},
            "potential_layers": {"type": "integer", "scalingType": "linear", "min": 0, "max": 5},
            "encoder_layers": {"type": "integer", "scalingType": "linear", "min": 1, "max": 5},
            # constant parameters
            "input_dim": {"type": "discrete", "values": [4]},
            "z_dim": {"type": "discrete", "values": [4]},
            "encoder_dropout_rate": {"type": "discrete", "values": [0.1]},
            "dissipative": {"type": "discrete", "values": [False]},
            "learn_kinetic": {"type": "discrete", "values": [False]},
            "dt": {"type": "discrete", "values": [0.1]},
            "discretization": {"type": "discrete", "values": [3]},
            "epochs": {"type": "discrete", "values": [10]},
            "batch_size": {"type": "discrete", "values": [256]},
            "learning_rate": {"type": "discrete", "values": [1e-3]},
            "beta1": {"type": "discrete", "values": [0.96]},
            "beta2": {"type": "discrete", "values": [0.999]},
            "clip_norm": {"type": "discrete", "values": [10.]},
            "lr_decay": {"type": "discrete", "values": [0.99996]},
            "weight_decay": {"type": "discrete", "values": [1e-3]},
            "annealing_epochs": {"type": "discrete", "values": [2]},
            "minimum_annealing_factor": {"type": "discrete", "values": [0.5]},
        },
        "trials": 1,
    }

    opt = comet_ml.Optimizer(config, api_key="Bm8mJ7xbMDa77te70th8PNcT8")
    param_sweep = ["emission_dim", "emission_layers", "transmission_dim", "potential_hidden", "potential_layers", "encoder_layers"]

    file_path = os.path.join(cfg.root_path, "sweeps", "2dof.txt")

    if not os.path.exists(file_path):
        os.mknod(file_path)

    for experiment in opt.get_experiments(project_name="2dof"):

        param_line = {par: experiment.get_parameter(par) for par in param_sweep}

        # Build the model:
        model = build_model_graph(experiment)

        # Train it:
        val_svi = train(experiment, model, loader_train)

        # How well did it do?
        final_loss, final_mse = evaluate(experiment, model, val_svi, loader_test, states_normalize, observations_normalize)
        param_line["mse"] = final_mse
        param_line["loss"] = final_loss

        result = json.dumps(param_line) + "\n"

        # write to sweep file
        with open(file_path, "a") as myfile:
            myfile.write(result)

        # free memory
        del model, val_svi, final_loss, final_mse

        # Optionally, end the experiment:
        experiment.end()
