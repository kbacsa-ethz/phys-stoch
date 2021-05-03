import torch
import torch.nn as nn


def init_xavier(model, retrain_seed):
    torch.manual_seed(retrain_seed)

    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain('relu')
            torch.nn.init.xavier_uniform_(m.weight, gain=g)
            # torch.nn.init.xavier_normal_(m.weight, gain=g)
            m.bias.data.fill_(0)

    model.apply(init_weights)


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


def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def first_order(y, x, create_graph=False):
    return torch.diagonal(jacobian(y, x, create_graph=create_graph))


def partial_grad(y, x, create_graph=False):
    grad_y = torch.zeros_like(y)
    grad_x = torch.zeros_like(x)
    for i in range(len(y)):
        grad_y[i] = 1.
        gradient, = torch.autograd.grad(y, x, grad_y, retain_graph=True, create_graph=create_graph)
        grad_x[i] = gradient[i]
        grad_y[i] = 0.
    return grad_x
