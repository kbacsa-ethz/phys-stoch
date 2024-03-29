import torch
import torch.nn as nn
from numpy.fft import rfft, irfft, rfftfreq
import numpy as np
from math import sqrt


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def data_path_from_config(cfg):
    data_path = '/'.join([
        cfg['System']['M'].replace('.', '-').replace(',', '_'),
        cfg['System']['C'].replace('.', '-').replace(',', '_'),
        cfg['System']['K'].replace('.', '-').replace(',', '_'),
        cfg['Forces']['Type'].replace('.', '-').replace(',', '_'),
        cfg['Forces']['Inputs'].replace('.', '-').replace(',', '_'),
        cfg['Simulation']['Noise'].replace('.', '-').replace(',', '_'),
    ]
    )
    data_path = cfg['System']['Name']
    return data_path


def init_xavier(model, retrain_seed):
    torch.manual_seed(retrain_seed)

    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad:
            g = nn.init.calculate_gain('relu')
            torch.nn.init.xavier_uniform_(m.weight, gain=g)
            # torch.nn.init.xavier_normal_(m.weight, gain=g)
            #m.bias.data.fill_(0)

    model.apply(init_weights)


# Triangular init
def tril_init(m):
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            m.weight.copy_(torch.tril(m.weight))


# Zero out gradients
def get_zero_grad_hook(mask):
    def hook(grad):
        return grad * mask
    return hook


def fill_triangular(x, upper=False):
    """
    ref : https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/linalg.py
    """

    m = x.size(0)
    n = int(sqrt(.25 + 2 * m) - .5)
    x_tail = x[(m - (n**2 - m)):]

    if upper:
        x_mat = torch.cat([x, torch.flip(x_tail, [-1])], 0).reshape([n, n])
        x_out = torch.triu(x_mat)
    else:
        x_mat = torch.cat([x_tail, torch.flip(x, [-1])], 0).reshape([n, n])
        x_out = torch.tril(x_mat)

    return x_out


def phase_shift(iptsignal, angle, dt):
    """Perform phase shift of arbitary angle

    Author: Xiao Xiao, https://github.com/SeisPider

    Parameter
    =========
    iptsignal : numpy.array
        input signal
    angle : float
        angle to shift signal, in degree
    dt : float
        time step

    """
    # Resolve the signal's fourier spectrum
    spec = rfft(iptsignal)
    freq = rfftfreq(iptsignal.size, d=dt)

    # Perform phase shift in freqeuency domain
    spec *= np.exp(1.0j * np.deg2rad(angle))

    # Inverse FFT back to time domain
    phaseshift = irfft(spec, n=len(iptsignal))
    return phaseshift


# this function takes a torch mini-batch and reverses each sequence
# (w.r.t. the temporal axis, i.e. axis=1).
def reverse_sequences(mini_batch, seq_lengths):
    reversed_mini_batch = torch.zeros_like(mini_batch)
    for b in range(mini_batch.size(0)):
        T = seq_lengths[b]
        time_slice = torch.arange(T - 1, -1, -1, device=mini_batch.device)
        reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence

    return reversed_mini_batch


# this function takes the hidden state as output by the PyTorch rnn and
# unpacks it it; it also reverses each sequence temporally
def pad_and_reverse(rnn_output, seq_lengths):
    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    reversed_output = reverse_sequences(rnn_output, seq_lengths)
    return reversed_output
