import matplotlib.pyplot as plt
from comet_ml import Experiment

import os
import argparse
import configparser
import json
from datetime import datetime
from pathlib import Path
import torch
from tqdm import tqdm
import matplotlib as mpl
import pandas as pd
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
from models import Emitter, GatedTransition, Combiner, RNNEncoder, ODEEncoder, SymplecticODEEncoder
from dmm import DMM
from utils import data_path_from_config
from plot_utils import *


# saves the model and optimizer states to disk
def save_checkpoint(model, optim, epoch, loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.get_state(),
        'loss': loss,
    }, os.path.join(save_path, 'checkpoint.pth'))


def train(cfg):
    hyper_params = vars(cfg)
    experiment = Experiment(project_name="phys-stoch", api_key="Bm8mJ7xbMDa77te70th8PNcT8", disabled=not cfg.comet)
    experiment.log_parameters(hyper_params)

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

    exp_name = data_path_from_config(config)
    obs_idx = list(map(int, config['Simulation']['Observations'].split(',')))

    # create experiment directory and save config
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    save_path = os.path.join(run_dir, exp_name, dt_string)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(save_path, 'config.txt'), 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

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
    val_idx = indexes[int(0.8 * len(indexes)):int(0.9 * len(indexes))]
    test_idx = indexes[int(0.9 * len(indexes)):]

    train_dataset = TrajectoryDataset(states_windowed[train_idx], observations_windowed[train_idx],
                                      forces_windowed[train_idx], obs_idx,
                                      cfg.seq_len + 1)
    val_dataset = TrajectoryDataset(states_windowed[val_idx], observations_windowed[val_idx], forces_windowed[val_idx],
                                    obs_idx, cfg.seq_len + 1)
    test_dataset = TrajectoryDataset(states_windowed[test_idx], observations_windowed[test_idx],
                                     forces_windowed[test_idx], obs_idx, cfg.seq_len + 1)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                               num_workers=cfg.nproc)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True,
                                             num_workers=cfg.nproc)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # free memory to avoid crash
    # states = None
    # observations = None
    # forces = None
    states_windowed = None
    observations_windowed = None
    forces_windowed = None

    # modules
    input_dim = len(obs_idx)
    z_dim = int(states.shape[-1] * 2 / 3)
    emitter = Emitter(input_dim, z_dim, cfg.emission_dim, cfg.emission_layers)
    transition = GatedTransition(z_dim, cfg.transmission_dim)
    combiner = Combiner(z_dim, z_dim)
    encoder = SymplecticODEEncoder(input_dim, z_dim, cfg.potential_hidden, cfg.potential_layers,
                                   non_linearity='relu', batch_first=True,
                                   rnn_layers=cfg.encoder_layers, dropout=cfg.encoder_dropout_rate,
                                   integrator=cfg.symplectic_integrator, dissipative=cfg.dissipative,
                                   learn_kinetic=cfg.learn_kinetic,
                                   dt=cfg.dt, discretization=cfg.discretization)

    # create model
    vae = DMM(emitter, transition, combiner, encoder, z_dim,
              (cfg.encoder_layers, cfg.batch_size, z_dim))
    vae.to(device)

    # setup optimizer
    adam_params = {"lr": cfg.learning_rate,
                   "betas": (cfg.beta1, cfg.beta2),
                   "clip_norm": cfg.clip_norm, "lrd": cfg.lr_decay,
                   "weight_decay": cfg.weight_decay}
    adam = ClippedAdam(adam_params)
    elbo = Trace_ELBO()
    svi = SVI(vae.model, vae.guide, adam, loss=elbo)

    with experiment.train():
        global_step = 0
        for epoch in range(cfg.num_epochs):
            epoch_loss = 0
            for which_batch, sample in enumerate(tqdm(train_loader)):
                if cfg.annealing_epochs > 0 and epoch < cfg.annealing_epochs:
                    # compute the KL annealing factor approriate for the current mini-batch in the current epoch
                    min_af = cfg.minimum_annealing_factor
                    annealing_factor = min_af + (1.0 - min_af) * \
                                       (float(which_batch + epoch * len(train_dataset) // cfg.batch_size + 1) /
                                        float(cfg.annealing_epochs * len(train_dataset) // cfg.batch_size))
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
                batch_loss = loss / cfg.batch_size
                experiment.log_metric("training_loss", batch_loss, step=global_step)
                optim_state = svi.optim.get_state()
                batch_lr = optim_state[next(iter(optim_state))]['param_groups'][0]['lr']
                experiment.log_metric("learning_rate", batch_lr, step=global_step)
                experiment.log_metric("C_rank", torch.linalg.matrix_rank(vae.emitter.hidden_to_loc.weight),
                                      step=global_step)
                if cfg.debug:
                    break

            epoch_loss /= len(train_dataset)
            print("Mean training loss at epoch {} is {}".format(epoch, epoch_loss))

            if not epoch % cfg.validation_freq:
                vae.eval()
                val_epoch_loss = 0
                for sample in tqdm(val_loader):
                    mini_batch = sample['obs'].float().to(device)
                    mini_batch_mask = torch.ones([mini_batch.size(0), mini_batch.size(1)]).to(device)

                    # do an actual gradient step
                    val_epoch_loss += svi.evaluate_loss(mini_batch, mini_batch_mask)
                    if cfg.debug:
                        break

                # record loss and save
                val_epoch_loss /= len(val_dataset)
                experiment.log_metric("validation_loss", val_epoch_loss, step=global_step)
                print("Mean validation loss at epoch {} is {}".format(epoch, val_epoch_loss))
                save_checkpoint(vae, svi.optim, epoch, val_epoch_loss, save_path)
                experiment.log_model("MODEL", os.path.join(save_path, "checkpoint.pth"))

                # Zhilu plot
                n_re = 0
                n_len = cfg.seq_len * 10
                sample = np.expand_dims(observations_normalize[n_re], axis=0)
                sample = torch.from_numpy(sample[:, : n_len + 1, :]).float()
                sample = sample.to(device)
                Z, Z_gen, Z_gen_scale, Obs, Obs_scale = vae.reconstruction(sample)

                fig = plot_emd(Z[0].detach().numpy(), debug=cfg.debug)
                experiment.log_figure(figure=fig, figure_name="HHT_{:02d}".format(epoch))

                ground_truth = np.expand_dims(states_normalize[n_re], axis=0)
                ground_truth = torch.from_numpy(ground_truth[:, : n_len, obs_idx]).float()
                ground_truth = ground_truth.to(device)
                mse_loss = ((ground_truth - Obs) ** 2).mean().item()
                experiment.log_metric("mse_loss", mse_loss, step=global_step)
                print("Mean error is {}".format(mse_loss))

                # unormalize for plots
                Z = Z.detach().numpy() * states_std[..., :z_dim] + states_mean[..., :z_dim]
                Z_gen = Z_gen.detach().numpy() * states_std[..., :z_dim] + states_mean[..., :z_dim]
                Z_gen_scale = Z_gen_scale.detach().numpy() * states_std[..., :z_dim] + states_mean[..., :z_dim]
                Obs = Obs.detach().numpy() * obs_std + obs_mean
                Obs_scale = Obs_scale.detach().numpy() * obs_std + obs_mean

                # TODO make this for any number of states
                # TODO get force derivatives
                q = Z[n_re, :, :z_dim // 2]
                qd = Z[n_re, :, z_dim // 2:]
                qdot = qd
                qdot = qdot[..., None]

                m = np.eye(z_dim // 2)
                latent_kinetic = 0.5 * np.matmul(np.transpose(qdot, axes=[0, 2, 1]), np.matmul(m, qdot))
                latent_kinetic = latent_kinetic.flatten()

                time_length = len(q)
                t_vec = torch.arange(1, time_length + 1) * cfg.dt

                if cfg.dissipative:
                    input_tensor = torch.cat([t_vec.float().unsqueeze(1), torch.from_numpy(q).float()], dim=1)
                else:
                    # input_tensor = torch.cat([torch.from_numpy(q).float(), torch.from_numpy(qd).float()], dim=1)
                    input_tensor = torch.from_numpy(q).float()

                latent_potential = vae.encoder.latent_func.energy(t_vec, input_tensor).detach().numpy()
                # latent_potential = vae.encoder.latent_func(t_vec, input_tensor).detach().numpy().sum(axis=1)
                latent_potential /= np.abs(latent_potential).max()

                # phase portrait
                fig, saved_phases = phase_plot(
                    pred_pos=q,
                    pred_vec=qd,
                    grnd_pos=states_normalize[n_re, :, :z_dim // 2],
                    grnd_vec=states_normalize[n_re, :, z_dim // 2:],
                    title="Phase",
                    debug=cfg.debug
                )

                for name, array in zip(['normal', 'cross', 'cross_inverted', 'inverted'], saved_phases):
                    experiment.log_table("{}_{}.csv".format(name, epoch), pd.DataFrame(array))

                experiment.log_figure(figure=fig, figure_name="phase_{:02d}".format(epoch))

                fig = simple_plot(
                    x_axis=t_vec,
                    values=[latent_kinetic,
                            energy[n_re, :time_length, 0],
                            latent_potential,
                            energy[n_re, :time_length, 1]],
                    max_t=100,
                    names=["learned kinetic", "true kinetic", "learned potential", "true potential"],
                    title="Energy",
                    normalized=True,
                    debug=cfg.debug
                )

                experiment.log_figure(figure=fig, figure_name="energy_{:02d}".format(epoch))

                # autonomous case
                z_true = states[..., :z_dim]
                Ylabels = ["u_" + str(i) for i in range(z_dim // 2)] + ["udot_" + str(i) for i in
                                                                        range(z_dim // 2)]
                fig = grid_plot(
                    x_axis=t_vec,
                    values=[z_true, Z_gen],
                    uncertainty=[None, Z_gen_scale],
                    max_1=n_re,
                    max_2=n_len,
                    n_plots=z_dim,
                    names=["reference", "generative model"],
                    title="Learned Latent Space - Training epoch =" + " " + str(epoch),
                    y_label=Ylabels,
                    debug=cfg.debug
                )
                experiment.log_figure(figure=fig, figure_name="latent_{:02d}".format(epoch))

                total_labels = ["u_" + str(i) for i in range(z_dim // 2)] + ["udot_" + str(i) for i in
                                                                             range(z_dim // 2)] + ["uddot_" + str(i) for
                                                                                                   i in
                                                                                                   range(z_dim // 2)]
                obs_labels = [total_labels[i] for i in obs_idx]
                fig = grid_plot(
                    x_axis=t_vec,
                    values=[Obs, observations],
                    uncertainty=[Obs_scale, None],
                    max_1=n_re,
                    max_2=n_len,
                    n_plots=input_dim,
                    names=["generated observations", "true observations"],
                    title='Observations - Training epoch =' + "" + str(epoch),
                    y_label=obs_labels,
                    debug=cfg.debug
                )
                experiment.log_figure(figure=fig, figure_name="observations_{:02d}".format(epoch))

                fig = matrix_plot(
                    matrix=vae.emitter.hidden_to_loc.weight.detach().numpy(),
                    title="Emission matrix at epoch = " + str(epoch),
                    debug=cfg.debug
                )
                experiment.log_figure(figure=fig, figure_name="c_mat_{:02d}".format(epoch))

                A = vae.trans.lin_proposed_mean_z_to_z.weight.detach().numpy()
                fig = matrix_plot(
                    matrix=A,
                    title="Transmission matrix at epoch = " + str(epoch),
                    debug=cfg.debug
                )
                experiment.log_figure(figure=fig, figure_name="a_mat_{:02d}".format(epoch))

                mass = vae.encoder.latent_func.m_1.data
                for i in range(z_dim // 2):
                    experiment.log_metric("mass_{}".format(i), mass[i], step=global_step)

                vae.train()

    return mse_loss


if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--config-path', type=str, default='config/2springmass_duffing_free.ini')
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
    parser.add_argument('-n', '--num-epochs', type=int, default=10)
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

    train(args)
