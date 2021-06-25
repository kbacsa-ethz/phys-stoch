from comet_ml import Experiment

import os
import argparse
import configparser
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
from utils import data_path_from_config, tril_init, get_zero_grad_hook

import matplotlib as mpl
import matplotlib.pyplot as plt


# saves the model and optimizer states to disk
def save_checkpoint(model, optim, epoch, loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.get_state(),
        'loss': loss,
    }, os.path.join(save_path, 'checkpoint.pth'))


def main(cfg):
    # add DLSC parameters like seed
    seed = 42
    torch.manual_seed(seed)

    hyper_params = {
        "seed": seed,
        "sequence_length": cfg.seq_len,
        "input_dim": cfg.input_dim,
        "z_dim": cfg.z_dim,
        "emission_dim": cfg.emission_dim,
        "emission_layers": cfg.emission_layers,
        "transmission_dim": cfg.transmission_dim,
        "encoder_dim": cfg.encoder_dim,
        "encoder_layers": cfg.encoder_layers,
        "batch_size": cfg.batch_size,
        "num_epochs": cfg.num_epochs,
        "learning_rate": cfg.learning_rate
    }

    experiment = Experiment(project_name="phys-stoch", api_key="Bm8mJ7xbMDa77te70th8PNcT8", disabled=not args.comet)
    experiment.log_parameters(hyper_params)

    if args.headless:
        mpl.use('Agg')  # if you are on a headless machine
    else:
        mpl.use('TkAgg')

    # use gpu
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    run_dir = os.path.join(cfg.root_path, 'experiments')

    config = configparser.ConfigParser()
    config.read(os.path.join(args.root_path, cfg.config))

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

    # normalize
    scaler = StandardScaler()

    n_obs, n_t, n_f = states.shape
    scaler.fit(np.reshape(states, [n_obs*n_t, n_f]))

    states_normalize = np.reshape(scaler.transform(np.reshape(states, [n_obs*n_t, n_f])), [n_obs, n_t, n_f])
    observations_normalize = np.zeros_like(observations)
    for idx, obs in enumerate(obs_idx):
        observations_normalize[idx] = (observations[idx] - scaler.mean_[obs]) * scaler.scale_[obs]

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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # free memory to avoid crash
    # states = None
    # observations = None
    # forces = None
    states_windowed = None
    observations_windowed = None
    forces_windowed = None

    # modules
    emitter = Emitter(cfg.input_dim, cfg.z_dim, cfg.emission_dim, cfg.emission_layers)

    # force triangular structure
    emitter.apply(tril_init)
    mask = torch.tril(torch.ones_like(emitter.hidden_to_loc.weight))
    # Register with hook
    emitter.hidden_to_loc.weight.register_hook(get_zero_grad_hook(mask))

    transition = GatedTransition(cfg.z_dim, cfg.transmission_dim)

    """
    # force triangular structure
    transition.apply(tril_init)
    mask = torch.tril(torch.ones_like(transition.lin_proposed_mean_z_to_z.weight))
    transition.lin_proposed_mean_z_to_z.weight.register_hook(get_zero_grad_hook(mask))
    """

    combiner = Combiner(cfg.z_dim, cfg.encoder_dim)
    encoder = SymplecticODEEncoder(cfg.input_dim, cfg.encoder_dim, cfg.potential_hidden, cfg.potential_layers,
                                   non_linearity='relu', batch_first=True,
                                   rnn_layers=cfg.encoder_layers, dropout=cfg.encoder_dropout_rate,
                                   dt=cfg.dt, discretization=cfg.discretization)

    # create model
    vae = DMM(emitter, transition, combiner, encoder, cfg.z_dim,
              (cfg.encoder_layers, cfg.batch_size, cfg.encoder_dim))
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
                #break

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
                    #break

                # record loss and save
                val_epoch_loss /= len(val_dataset)
                experiment.log_metric("validation_loss", val_epoch_loss, step=global_step)
                print("Mean validation loss at epoch {} is {}".format(epoch, val_epoch_loss))
                save_checkpoint(vae, svi.optim, epoch, val_epoch_loss, save_path)

                # Zhilu plot
                n_re = 0
                n_len = cfg.seq_len * 10
                sample = np.expand_dims(observations_normalize[n_re], axis=0)
                sample = torch.from_numpy(sample[:, : n_len + 1, :]).float()
                sample = sample.to(device)
                Z, Z_gen, Z_gen_scale, Obs, Obs_scale = vae.reconstruction(sample)

                # TODO make this for any number of states
                # TODO get force derivatives
                q = Z[n_re, :, :cfg.z_dim // 2].data
                qd = Z[n_re, :, cfg.z_dim // 2:].data

                m = torch.eye(cfg.z_dim // 2)

                # e_kin = torch.zeros()
                # for t in range(q.size(1)):

                time_length = q.size(0)
                e_kin = torch.zeros(time_length, 1)
                for t in range(time_length):
                    e_kin[t] = torch.dot(qd[t, :], torch.matmul(m, qd[t, :].unsqueeze(-1)).squeeze(-1))

                t_vec = torch.arange(1, time_length)
                e_pot = vae.encoder.latent_func(t_vec, torch.cat([q, qd], dim=1))

                fig0 = plt.figure(figsize=(16, 7))
                plt.plot(e_kin)
                plt.plot(e_pot.detach())
                #plt.show()
                experiment.log_figure(figure=fig0, figure_name="energy_{:02d}".format(epoch))

                # autonomous case
                z_true = states_normalize[..., :cfg.z_dim]
                Ylabels = ["u_" + str(i) for i in range(cfg.z_dim // 2)] + ["udot_" + str(i) for i in
                                                                            range(cfg.z_dim // 2)]

                fig1 = plt.figure(figsize=(16, 7))
                plt.ioff()
                for i in range(cfg.z_dim):
                    ax = plt.subplot(cfg.z_dim // 2, cfg.z_dim // (cfg.z_dim // 2), i + 1)
                    plt.plot(z_true[n_re, :n_len, i], color="silver", lw=2.5, label="reference")
                    plt.plot(Z[n_re, :, i].data, label="inference")
                    plt.plot(Z_gen[n_re, :, i].data, label="generative model")

                    # plot observations if needed
                    if i in obs_idx:
                        plt.plot(Obs[n_re, :n_len, i].data, label="generated observations")
                        plt.plot(observations[n_re, :n_len, i], label="observations")
                        lower_bound = Obs[n_re, :n_len, i].data - Obs_scale[n_re, :n_len, i].data
                        upper_bound = Obs[n_re, :n_len, i].data + Obs_scale[n_re, :n_len, i].data
                        ax.fill_between(np.arange(0, n_len, 1), lower_bound, upper_bound,
                                        facecolor='yellow', alpha=0.5,
                                        label='1 sigma range')
                    plt.legend()
                    plt.xlabel("$k$")
                    plt.ylabel(Ylabels[i])

                fig1.suptitle('Learned Latent Space - Training epoch =' + "" + str(epoch))
                plt.tight_layout()
                # plt.show()
                experiment.log_figure(figure=fig1, figure_name="latent_{:02d}".format(epoch))

                Ylabels = ["u_" + str(i) for i in range(cfg.z_dim // 2)] + ["uddot_" + str(i) for i in
                                                                            range(cfg.z_dim // 2)]
                fig2 = plt.figure(figsize=(16, 7))
                plt.ioff()
                for i in range(cfg.input_dim):
                    ax = plt.subplot(cfg.input_dim // 2, cfg.input_dim // (cfg.input_dim // 2), i + 1)

                    plt.plot(Obs[n_re, :n_len, i].data, label="generated observations")
                    plt.plot(observations[n_re, :n_len, i], label="observations")
                    lower_bound = Obs[n_re, :n_len, i].data - Obs_scale[n_re, :n_len, i].data
                    upper_bound = Obs[n_re, :n_len, i].data + Obs_scale[n_re, :n_len, i].data
                    ax.fill_between(np.arange(0, n_len, 1), lower_bound, upper_bound,
                                    facecolor='yellow', alpha=0.5,
                                    label='1 sigma range')
                    plt.legend()
                    plt.xlabel("$k$")
                    plt.ylabel(Ylabels[i])

                fig2.suptitle('Observations - Training epoch =' + "" + str(epoch))
                plt.tight_layout()
                # plt.show()
                experiment.log_figure(figure=fig2, figure_name="observations_{:02d}".format(epoch))

                fig3 = plt.figure(figsize=(16, 7))
                plt.ioff()
                c_mat = vae.emitter.hidden_to_loc.weight.detach().numpy()
                plt.imshow(c_mat)
                for i in np.arange(np.shape(c_mat)[0]):  # over all rows of count
                    for j in np.arange(np.shape(c_mat)[1]):  # over all cols of count
                        text = plt.text(j, i, str(round(c_mat[i, j], 2)), ha="center", va="center", color="r")
                # plt.show()
                experiment.log_figure(figure=fig3, figure_name="c_mat_{:02d}".format(epoch))

                fig4 = plt.figure(figsize=(16, 7))
                plt.ioff()
                a_mat = vae.trans.lin_proposed_mean_z_to_z.weight.detach().numpy()
                plt.imshow(a_mat)
                for i in np.arange(np.shape(a_mat)[0]):  # over all rows of count
                    for j in np.arange(np.shape(a_mat)[1]):  # over all cols of count
                        text = plt.text(j, i, str(round(a_mat[i, j], 2)), ha="center", va="center", color="r")
                # plt.show()
                experiment.log_figure(figure=fig3, figure_name="a_mat_{:02d}".format(epoch))

                vae.train()

    return 0


if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--config', type=str, default='config/2springmass_free.ini')
    parser.add_argument('-in', '--input-dim', type=int, default=4)
    parser.add_argument('-z', '--z-dim', type=int, default=4)
    parser.add_argument('-e', '--emission-dim', type=int, default=16)
    parser.add_argument('-ne', '--emission-layers', type=int, default=0)
    parser.add_argument('-tr', '--transmission-dim', type=int, default=32)
    parser.add_argument('-ph', '--potential-hidden', type=int, default=60)
    parser.add_argument('-pl', '--potential-layers', type=int, default=0)
    parser.add_argument('-enc', '--encoder-dim', type=int, default=4)
    parser.add_argument('-nenc', '--encoder-layers', type=int, default=2)
    parser.add_argument('-dt', '--dt', type=float, default=0.1)
    parser.add_argument('-disc', '--discretization', type=int, default=3)
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
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--comet', action='store_true')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    main(args)
