import numpy as np
import torch

from torch.utils.data.sampler import SubsetRandomSampler
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    TraceTMC_ELBO,
    config_enumerate,
)
from pyro.optim import ClippedAdam

from phys_data import TrajectoryDataset
from models import Emitter, GatedTransition, Combiner, RNNEncoder
from dmm import DMM
from utils import init_xavier



def main():
    # add DLSC parameters like seed
    seed = 42
    shuffle_dataset = True
    validation_split = 0.1
    batch_size = 10
    n_epochs = 10
    annealing_epochs = 3

    # load dataset
    data_path = './data'
    experiment_name = 'springmass_sinusoidal'
    obs_idx = [0]
    sequence_length = 50

    # parameters
    obs_dim = 1
    z_dim = 3  # q, qdot, force
    hidden_emitter = 11
    layers_emitter = 2
    hidden_transition = 12
    hidden_encoder = 13
    layers_encoder = 2

    dataset = TrajectoryDataset(data_path, experiment_name, obs_idx, sequence_length)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    # modules
    emitter = Emitter(obs_dim, z_dim, hidden_emitter, layers_emitter)
    transition = GatedTransition(z_dim, hidden_transition)
    combiner = Combiner(z_dim, hidden_encoder)
    encoder = RNNEncoder(obs_dim, hidden_encoder,
                         non_linearity='relu', batch_first=True, num_layers=layers_encoder, dropout=0.1, seq_len=50)

    vae = DMM(emitter, transition, combiner, encoder, z_dim, (layers_encoder, batch_size, hidden_encoder))

    init_xavier(vae, seed)

    # add optimizer and SVI
    # setup optimizer
    adam_params = {"lr": 1e-3, "betas": (0.96, 0.99),
                   "clip_norm": 10.0, "lrd": 0.97,
                   "weight_decay": 2.0}
    adam = ClippedAdam(adam_params)
    # training loop

    elbo = Trace_ELBO()
    #dmm_guide = config_enumerate(vae.guide, default="parallel", num_samples=17, expand=False)
    svi = SVI(vae.model, vae.guide, adam, loss=elbo)

    for epoch in range(n_epochs):

        for sample in train_loader:
            mini_batch = sample['obs'].float()
            mini_batch_mask = torch.ones([mini_batch.size(0), mini_batch.size(1)])

            # do an actual gradient step
            loss = svi.step(mini_batch, mini_batch_mask)

            print(loss)


    return 0


if __name__ == '__main__':
    # parse config
    main()
