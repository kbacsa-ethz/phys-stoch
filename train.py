import os
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    TraceTMC_ELBO,
    config_enumerate,
)
from pyro.optim import ClippedAdam

# visualization
from tensorboardX import SummaryWriter
import wandb

from phys_data import TrajectoryDataset
from models import Emitter, GatedTransition, Combiner, RNNEncoder
from dmm import DMM
from utils import init_xavier


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

    # load dataset
    run_dir = './experiments'
    obs_idx = [0]

    # create experiment directory and save config
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    save_path = os.path.join(run_dir, cfg.exp_name, dt_string)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(save_path, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    writer = SummaryWriter(logdir=save_path)
    wandb.init(project="Deep-markov-model", dir=save_path, config=cfg, sync_tensorboard=True)

    states = np.load(os.path.join(args.data_dir, args.exp_name, 'state.npy'))
    observations = np.load(os.path.join(args.data_dir, args.exp_name, 'obs.npy'))
    forces = np.load(os.path.join(args.data_dir, args.exp_name, 'force.npy'))

    n_exp = states.shape[0]

    # 80/10/10 split by default
    indexes = np.arange(n_exp)
    train_idx = indexes[:int(0.8 * len(indexes))]
    val_idx = indexes[int(0.8 * len(indexes)):int(0.9 * len(indexes))]
    test_idx = indexes[int(0.9 * len(indexes)):]

    train_dataset = TrajectoryDataset(states[train_idx], observations[train_idx], forces[train_idx], obs_idx,
                                      args.seq_len)
    val_dataset = TrajectoryDataset(states[val_idx], observations[val_idx], forces[val_idx], obs_idx, args.seq_len)
    test_dataset = TrajectoryDataset(states[test_idx], observations[test_idx], forces[test_idx], obs_idx, args.seq_len)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # modules
    emitter = Emitter(args.input_dim, args.z_dim, args.emission_dim, args.emission_layers)
    transition = GatedTransition(args.z_dim, args.transmission_dim)
    combiner = Combiner(args.z_dim, args.encoder_dim)
    encoder = RNNEncoder(args.input_dim, args.encoder_dim,
                         non_linearity='relu', batch_first=True, num_layers=args.encoder_layers,
                         dropout=args.encoder_dropout_rate, seq_len=args.seq_len)

    # create model
    vae = DMM(emitter, transition, combiner, encoder, args.z_dim,
              (args.encoder_layers, args.batch_size, args.encoder_dim))
    wandb.watch(vae, log_freq=args.validation_freq)
    init_xavier(vae, seed)

    # setup optimizer
    adam_params = {"lr": args.learning_rate,
                   "betas": (args.beta1, args.beta2),
                   "clip_norm": args.clip_norm, "lrd": args.lr_decay,
                   "weight_decay": args.weight_decay}
    adam = ClippedAdam(adam_params)
    elbo = Trace_ELBO()
    svi = SVI(vae.model, vae.guide, adam, loss=elbo)

    global_step = 0
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        for which_batch, sample in enumerate(tqdm(train_loader)):
            if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
                # compute the KL annealing factor approriate for the current mini-batch in the current epoch
                min_af = args.minimum_annealing_factor
                annealing_factor = min_af + (1.0 - min_af) * \
                                   (float(which_batch + epoch * len(train_dataset) // args.batch_size + 1) /
                                    float(args.annealing_epochs * len(train_dataset) // args.batch_size))
            else:
                # by default the KL annealing factor is unity
                annealing_factor = 1.0

            mini_batch = sample['obs'].float()
            mini_batch_mask = torch.ones(
                [mini_batch.size(0), mini_batch.size(1)])  # assumption that all sequences have the same length

            # do an actual gradient step
            loss = svi.step(mini_batch, mini_batch_mask, annealing_factor)
            epoch_loss += loss

            # record loss
            global_step += 1
            batch_loss = loss / args.batch_size
            wandb.log({"loss/training_loss": batch_loss, "global_step": global_step})
            writer.add_scalar('loss/training_loss', batch_loss, global_step)
            optim_state = svi.optim.get_state()
            batch_lr = optim_state[next(iter(optim_state))]['param_groups'][0]['lr']
            wandb.log({"loss/learning_rate": batch_lr, "global_step": global_step})
            writer.add_scalar('loss/learning_rate', batch_lr, global_step)

        epoch_loss /= len(train_dataset)
        print("Mean training loss at epoch {} is {}".format(epoch, epoch_loss))

        if not epoch % args.validation_freq:
            vae.eval()
            val_epoch_loss = 0
            for sample in tqdm(val_loader):
                mini_batch = sample['obs'].float()
                mini_batch_mask = torch.ones([mini_batch.size(0), mini_batch.size(1)])

                # do an actual gradient step
                val_epoch_loss += svi.evaluate_loss(mini_batch, mini_batch_mask)

            # record loss and save
            val_epoch_loss /= len(val_dataset)
            writer.add_scalar('loss/validation_loss', val_epoch_loss, global_step)
            wandb.log({"loss/validation_loss": val_epoch_loss, "global_step": global_step})
            print("Mean validation loss at epoch {} is {}".format(epoch, val_epoch_loss / len(val_dataset)))
            save_checkpoint(vae, svi.optim, epoch, val_epoch_loss, save_path)
            vae.train()

    return 0


if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--exp-name', type=str, default='springmass_sinusoidal')
    parser.add_argument('-in', '--input-dim', type=int, default=1)
    parser.add_argument('-z', '--z-dim', type=int, default=3)
    parser.add_argument('-e', '--emission-dim', type=int, default=8)
    parser.add_argument('-ne', '--emission-layers', type=int, default=1)
    parser.add_argument('-tr', '--transmission-dim', type=int, default=16)
    parser.add_argument('-enc', '--encoder-dim', type=int, default=48)
    parser.add_argument('-nenc', '--encoder-layers', type=int, default=3)
    parser.add_argument('-n', '--num-epochs', type=int, default=5000)
    parser.add_argument('-te', '--tuning-epochs', type=int, default=5000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('-b1', '--beta1', type=float, default=0.96)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip-norm', type=float, default=10.0)
    parser.add_argument('-lrd', '--lr-decay', type=float, default=0.99996)
    parser.add_argument('-wd', '--weight-decay', type=float, default=2.0)
    parser.add_argument('-bs', '--batch-size', type=int, default=20)
    parser.add_argument('-sq', '--seq-len', type=int, default=50)
    parser.add_argument('-ae', '--annealing-epochs', type=int, default=1000)
    parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.75)
    parser.add_argument('-rdr', '--encoder-dropout-rate', type=float, default=0.1)
    parser.add_argument('-iafs', '--num-iafs', type=int, default=0)
    parser.add_argument('-id', '--iaf-dim', type=int, default=100)
    parser.add_argument('-vf', '--validation-freq', type=int, default=100)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    main(args)
