import os
import numpy as np
import torch

import models
from phys_data import TrajectoryDataset
from torch.utils.data.dataloader import DataLoader
from models import Emitter, GatedTransition, Combiner, ODEEncoder
from dmm import DMM

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine


def main():

    x_dim = 2
    z_dim = 6

    data_dir = './data'
    exp_name = '2springmass_sinusoidal'

    states = np.load(os.path.join(data_dir, exp_name, 'state.npy'))
    observations = np.load(os.path.join(data_dir, exp_name, 'obs.npy'))
    forces = np.load(os.path.join(data_dir, exp_name, 'force.npy'))


    dataset = TrajectoryDataset(states, observations, forces, [1, 4], 50)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    example = next(iter(loader))

    x_in = example['obs']
    z_in = example['state'][:, 0, [0, 1, 2, 3]].float()

    # add 0 force
    z_in = torch.cat([z_in, torch.zeros([z_in.size(0), 2])], dim=-1)

    # state dim = 2
    # obs = 1
    emitter = Emitter(x_dim, z_dim, 16, 2)

    x_loc, x_scale = emitter(z_in)

    trans = GatedTransition(z_dim, 16)

    z_loc, z_scale = trans(z_in)

    """
    T_max = 50
    batch_size = 4
    with pyro.plate("z_minibatch", batch_size):
        for t in pyro.markov(range(1, T_max + 1)):
            with poutine.scale(scale=1.0):
                z_t = pyro.sample("z_%d" % t,
                                  dist.Normal(z_loc, z_scale)
                                  .to_event(1))

    """

    #encoder = models.RNNEncoder(2, 16, 'relu', True, 2, 0.0, 50)
    encoder = models.SymplecticODEEncoder(x_dim, z_dim, 20, 3, 'relu', True, 3, 0.0, 50, 0.1, 10)

    q_z_x = encoder(x_in.float())

    combiner = Combiner(z_dim, z_dim)
    vae = DMM(emitter, trans, combiner, encoder, z_dim, (2, 4, 16))
    mini_batch_mask = torch.ones(
        [x_in.size(0), x_in.size(1)])
    vae.model(x_in.float(), mini_batch_mask)
    vae.guide(x_in.float(), mini_batch_mask)

    return 0


if __name__ == '__main__':

    main()