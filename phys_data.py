import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, states, observations, forces, input_dofs, sequence_length):

        if len(observations) > 3:
            observations = np.reshape(observations, [-1, *observations.shape[2:]])
            states = np.reshape(states, [-1, *states.shape[2:]])
            forces = np.reshape(forces, [-1, *forces.shape[2:]])

        self.states = states
        self.observations = observations
        self.forces = forces

        self.seq_len = sequence_length
        self.n_exp, self.n_time, self.n_states = self.states.shape
        self.n_dof = self.n_states // 3
        self.input_dofs = input_dofs
        self.n_obs = self.observations.shape[-1]
        self.states = torch.from_numpy(
            np.reshape(self.states, [self.n_exp * self.n_time // sequence_length, sequence_length, self.n_states]))
        self.observations = torch.from_numpy(
            np.reshape(self.observations, [self.n_exp * self.n_time // sequence_length, sequence_length, self.n_obs]))
        self.forces = torch.from_numpy(
            np.reshape(self.forces, [self.n_exp * self.n_time // sequence_length, sequence_length, self.n_dof]))

    def __len__(self):
        return self.n_exp * self.n_time // self.seq_len

    def __getitem__(self, idx):
        state = self.states[idx]
        obs = self.observations[idx]
        force = self.forces[idx]
        sample = {'state': state, 'obs': obs, 'force': force}
        return sample