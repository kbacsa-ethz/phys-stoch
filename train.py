import os
import argparse
import configparser
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import matplotlib as mpl
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
from models import *
from dmm import DMM
from plot_utils import *


# saves the model and optimizer states to disk
def save_checkpoint(model, state_mu, state_sig, obs_mu, obs_sig, optim, epoch, loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.get_state(),
        'loss': loss,
        'state_mu': state_mu,
        'state_sig': state_sig,
        'obs_mu': obs_mu,
        'obs_sig': obs_sig,
    }, os.path.join(save_path, 'checkpoint.pth'))


def train(cfg):
    hyper_params = vars(cfg)

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

    exp_name = os.path.basename(cfg.config_path).split('.')[0]
    obs_idx = list(map(int, config['Simulation']['Observations'].split(',')))

    # create experiment directory and save config
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    save_path = os.path.join(run_dir, exp_name, dt_string)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(save_path, 'config.txt'), 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    model_name = cfg.config_path.split("/")[-1].split(".")[0]  # does not work on windows

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

    # Save normalization parameters
    print("states_mean: {}".format(states_mean))
    print("states_std: {}".format(states_std))
    print("obs_mean: {}".format(obs_mean))
    print("obs_std: {}".format(obs_std))

    # windowed dataset
    n_exp = states.shape[0]
    observations_windowed = []
    states_windowed = []
    forces_windowed = []
    for t in range(observations.shape[1] - cfg.seq_len):
        qqd_w = observations_normalize[:, t:t + cfg.seq_len + 1]
        observations_windowed.append(qqd_w[:, None])
        qqd_w = states_normalize[:, t:t + cfg.seq_len + 1]
        states_windowed.append(qqd_w[:, None])
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
    # normalize Obs for model input but keep unnormalized states for absolute MSE and error
    test_dataset = TrajectoryDataset(states[test_idx], observations_normalize[test_idx],
                                     forces[test_idx], obs_idx, states.shape[1])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                               num_workers=cfg.nproc)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True,
                                             num_workers=cfg.nproc)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # modules
    input_dim = len(obs_idx)
    z_dim = int(states.shape[-1] * 2 / 3)
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
                                       order=cfg.integrator_order, dissipative=True if cfg.dissipative == 'dissipative' else False,
                                       learn_kinetic=cfg.learn_kinetic,
                                       dt=cfg.dt, discretization=cfg.discretization)
    else:
        raise NotImplementedError

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
            optim_state = svi.optim.get_state()
            batch_lr = optim_state[next(iter(optim_state))]['param_groups'][0]['lr']

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

            # record loss and save
            val_epoch_loss /= len(val_dataset)
            print("Mean validation loss at epoch {} is {}".format(epoch, val_epoch_loss))
            save_checkpoint(vae, states_mean, states_std, obs_mean, obs_std, svi.optim, epoch, val_epoch_loss, save_path)

            mse = torch.zeros(len(test_dataset))
            error = torch.zeros(len(test_dataset))
            print("Testing on {} samples".format(len(test_dataset)))
            for idx, sample in tqdm(enumerate(test_loader)):
                n_len = cfg.seq_len * 10
                sample_obs = sample['obs'][:, : n_len + 1, :].float()
                sample_obs = sample_obs.to(device)
                Z, Z_gen, Z_gen_scale, Obs, Obs_scale = vae.reconstruction(sample_obs, 1)
                Obs = Obs.detach() * obs_std + obs_mean
                Obs_scale = Obs_scale.detach() * obs_std + obs_mean
                ground_truth = sample['state'][:, : n_len, obs_idx].float()
                ground_truth = ground_truth.to(device)
                mse[idx] = torch.abs(
                    (Obs - ground_truth) / (ground_truth + 1e-6)).mean().item()  # add 1e-6 to avoid inf value
                error[idx] = torch.logical_or(torch.lt(ground_truth, (Obs - 2 * Obs_scale)),
                                              torch.gt(ground_truth, (Obs + 2 * Obs_scale))).float().mean()

            print("Mean MSE is {}".format(mse.mean().item()))
            print("Mean error is {:.2%}".format(error.mean().item()))

            vae.train()
    return 0


if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--config-path', type=str, default='config/2_springmass_duffing_free_free_0,1,4,5.ini')
    parser.add_argument('-e', '--emission-dim', type=int, default=16)
    parser.add_argument('-ne', '--emission-layers', type=int, default=0)
    parser.add_argument('-tr', '--transmission-dim', type=int, default=32)
    parser.add_argument('-ph', '--potential-hidden', type=int, default=60)
    parser.add_argument('-pl', '--potential-layers', type=int, default=2)
    parser.add_argument('-tenc', '--encoder-type', type=str, default="symplectic_node")
    parser.add_argument('-nenc', '--encoder-layers', type=int, default=2)
    parser.add_argument('-ord', '--integrator-order', type=int, default=2)
    parser.add_argument('--dissipative', type=str, default="free")
    parser.add_argument('-dt', '--dt', type=float, default=0.01)
    parser.add_argument('-disc', '--discretization', type=int, default=3)
    parser.add_argument('-n', '--num-epochs', type=int, default=2)
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
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    train(args)
