import os
import numpy as np
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, root_dir, exp_dir, input_dofs, sequence_length):

        self.states = np.load(os.path.join(root_dir, exp_dir, 'state.npy'))
        self.observations = np.load(os.path.join(root_dir, exp_dir, 'obs.npy'))
        self.forces = np.load(os.path.join(root_dir, exp_dir, 'force.npy'))

        self.seq_len = sequence_length
        self.n_exp, self.n_time, self.n_states = self.states.shape
        self.n_dof = self.n_states // 3
        self.input_dofs = input_dofs
        self.n_obs = self.observations.shape[-1]
        self.states = np.reshape(self.states, [self.n_exp * self.n_time//sequence_length, sequence_length, self.n_states])
        self.observations = np.reshape(self.observations, [self.n_exp * self.n_time//sequence_length, sequence_length, self.n_obs])
        self.forces = np.reshape(self.forces, [self.n_exp * self.n_time//sequence_length, sequence_length, self.n_dof])

    def __len__(self):
        return self.n_exp * self.n_time // self.seq_len

    def __getitem__(self, idx):
        state = self.states[idx]
        obs = self.observations[idx]
        force = self.forces[idx]
        sample = {'state': state, 'obs': obs, 'force': force}
        return sample