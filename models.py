import torch
import torch.nn as nn
import utils

from torchdiffeq.torchdiffeq import odeint


class PosSemiDefLayer(nn.Module):

    def __init__(self, dim):
        super().__init__()

        num_w = int(dim * (dim + 1) / 2)
        self.l_param = nn.Parameter(torch.ones(num_w), requires_grad=True)
        self.weight = None

    def forward(self, x):
        l = utils.fill_triangular(self.l_param)
        w = torch.transpose(l, 0, 1) * l
        self.weight = w
        out = torch.nn.functional.linear(x, w)
        return out


class Emitter(nn.Module):
    """
    Parameterizes the observation likelihood p(x_t | z_t)
    """

    def __init__(self, input_dim, z_dim, emission_dim, h_layers):
        super().__init__()

        if h_layers == 0:
            self.hidden_to_loc = nn.Linear(z_dim, input_dim, bias=False)
            self.hidden_to_scale = nn.Linear(z_dim, input_dim, bias=False)
        else:
            self.hidden_to_loc = nn.Linear(emission_dim, input_dim, bias=False)
            self.hidden_to_scale = nn.Linear(emission_dim, input_dim, bias=False)

        self.linears = nn.ModuleList([])
        for layer in range(h_layers):
            if layer == 0:
                self.linears.append(nn.Linear(z_dim, emission_dim))
            else:
                self.linears.append(nn.Linear(emission_dim, emission_dim))

        """ Check if matrix is triangular
        self.hidden_to_loc.register_forward_hook(
            lambda layer, _, output: print({layer.weight})
        )
        """

        self.n_layers = len(self.linears)
        self.input_dim = input_dim
        self.emission_dim = emission_dim
        self.h_activation = nn.ReLU()
        self.e_activation = nn.Softplus()

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the distribution p(x_t|z_t)
        """
        x = z_t
        # hidden layers
        for layer in range(self.n_layers):
            x = self.h_activation(self.linears[layer](x))

        loc = self.hidden_to_loc(x)
        scale = self.e_activation(self.hidden_to_scale(x))
        return loc, scale


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability p(z_t | z_{t-1})
    See section 5 in the reference for comparison.
    """

    def __init__(self, z_dim, transition_dim):
        super().__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_z = PosSemiDefLayer(z_dim)
        #self.lin_proposed_mean_z_to_z = nn.Linear(z_dim, z_dim, bias=False)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.z_dim = z_dim

    def forward(self, z_t_1):
        """
        Given the latent z_{t-1} corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution p(z_t | z_{t-1})
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = self.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        proposed_mean = self.lin_proposed_mean_z_to_z(z_t_1)
        # assemble the actual mean used to sample z_t, which mixes
        # a linear transformation of z_{t-1} with the proposed mean
        # modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed
        # mean from above as input. the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        return loc, scale


class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the encoder
    """

    def __init__(self, z_dim, encoder_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, encoder_dim)
        self.lin_hidden_to_loc = nn.Linear(encoder_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(encoder_dim, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the encoder `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`
        """
        # combine the encoder hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        # return loc, scale which can be fed into Normal
        return loc, scale


class RNNEncoder(nn.Module):
    """
    Parameterizes `q(z_t | x_{t:T})`
    """

    def __init__(self, input_size, hidden_size, non_linearity, batch_first, num_layers, dropout, seq_len):
        super().__init__()

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, nonlinearity=non_linearity,
                          batch_first=batch_first, bidirectional=False, num_layers=num_layers, dropout=dropout)

        self.h_0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_layers = num_layers

    def forward(self, x):
        x_reversed = utils.reverse_sequences(x, [self.seq_len] * x.size(0))

        # do sequence packing
        x_reversed = nn.utils.rnn.pack_padded_sequence(x_reversed,
                                                       [self.seq_len] * x.size(0),
                                                       batch_first=True)

        h_0_contig = self.h_0.expand(self.num_layers, x.size(0), self.rnn.hidden_size).contiguous()
        rnn_output, _ = self.rnn(x_reversed, h_0_contig)
        rnn_output = utils.pad_and_reverse(rnn_output, [self.seq_len] * x.size(0))
        return rnn_output


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class ODEEncoder(nn.Module):
    """
    Parameterizes `q(z_t | x_{t:T})`
    """

    def __init__(self, input_size, z_dim, hidden_dim, n_layers, non_linearity, batch_first, rnn_layers, dropout, seq_len, dt, discretization):
        super().__init__()

        self.latent_func = LatentODEfunc(z_dim, hidden_dim)

        self.rnn = nn.RNN(input_size=input_size, hidden_size=z_dim, nonlinearity=non_linearity,
                          batch_first=batch_first, bidirectional=False, num_layers=rnn_layers, dropout=dropout)

        self.h_0 = nn.Parameter(torch.zeros(rnn_layers, 1, z_dim))
        self.rnn_layers = rnn_layers
        self.seq_len = seq_len
        self.time = torch.arange(0, dt, dt/discretization)

    def forward(self, x):
        x_reversed = utils.reverse_sequences(x, [self.seq_len] * x.size(0))

        # do sequence packing
        x_reversed = nn.utils.rnn.pack_padded_sequence(x_reversed,
                                                       [self.seq_len] * x.size(0),
                                                       batch_first=True)

        h_0_contig = self.h_0.expand(self.rnn_layers, x.size(0), self.rnn.hidden_size).contiguous()
        rnn_output, _ = self.rnn(x_reversed, h_0_contig)
        rnn_output = utils.pad_and_reverse(rnn_output, [self.seq_len] * x.size(0))

        ode_output = torch.zeros_like(rnn_output)
        ode_output[:, -1, :] = rnn_output[:, -1, :]
        for t in reversed(range(self.seq_len-1)):
            ode_output[:, t, :] = odeint(self.latent_func, rnn_output[:, t+1, :], self.time)[-1]
        return ode_output


class PotentialODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20, hlayers=0):
        super(PotentialODEfunc, self).__init__()
        self.tanh = nn.Tanh()

        self.linears = nn.ModuleList([])
        self.linears.append(nn.Linear(latent_dim, nhidden))

        for layer in range(hlayers):
            self.linears.append(nn.Linear(nhidden, nhidden))

        self.fc = nn.Linear(nhidden, 1)
        self.nlayers = len(self.linears)
        self.nfe = 0

    def forward(self, t, x):
        with torch.enable_grad():
            one = torch.tensor(1, dtype=torch.float32, device=x.device, requires_grad=True)
            x *= one
            out = x.clone()
            self.nfe += 1

            for layer in range(self.nlayers):
                out = self.tanh(self.linears[layer](out))

            out = self.fc(out)
            out = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
        return out


class GradPotentialODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20, hlayers=0):
        super(GradPotentialODEfunc, self).__init__()
        self.tanh = nn.Tanh()

        self.linears = nn.ModuleList([])
        self.linears.append(nn.Linear(latent_dim, nhidden))

        for layer in range(hlayers):
            self.linears.append(nn.Linear(nhidden, nhidden))

        self.fc = nn.Linear(nhidden, latent_dim)
        self.nlayers = len(self.linears)
        self.nfe = 0

    def forward(self, t, x):
        out = x.clone()
        for layer in range(self.nlayers):
            out = self.tanh(self.linears[layer](out))
        out = self.fc(out)
        return out


class SymplecticODEEncoder(nn.Module):
    """
    Parameterizes `q(z_t | x_{t:T})`
    """

    def __init__(self,
                 input_size, z_dim,
                 hidden_dim, n_layers, non_linearity, batch_first, rnn_layers, dropout,
                 integrator,
                 dt, discretization):
        super().__init__()

        self.latent_func = PotentialODEfunc(z_dim, hidden_dim, n_layers)

        self.rnn = nn.RNN(input_size=input_size, hidden_size=z_dim, nonlinearity=non_linearity,
                          batch_first=batch_first, bidirectional=False, num_layers=rnn_layers, dropout=dropout)

        self.h_0 = nn.Parameter(torch.zeros(rnn_layers, 1, z_dim))
        self.rnn_layers = rnn_layers
        self.integrator = integrator
        self.time = torch.arange(0, dt, dt/discretization)

    def forward(self, x, seq_len):
        x_reversed = utils.reverse_sequences(x, [seq_len] * x.size(0))

        # do sequence packing
        x_reversed = nn.utils.rnn.pack_padded_sequence(x_reversed,
                                                       [seq_len] * x.size(0),
                                                       batch_first=True)

        h_0_contig = self.h_0.expand(self.rnn_layers, x.size(0), self.rnn.hidden_size).contiguous()
        rnn_output, _ = self.rnn(x_reversed, h_0_contig)
        rnn_output = utils.pad_and_reverse(rnn_output, [seq_len] * x.size(0))

        ode_output = torch.zeros_like(rnn_output)
        ode_output[:, -1, :] = rnn_output[:, -1, :]
        for t in reversed(range(seq_len-1)):
            ode_output[:, t, :] = odeint(self.latent_func, rnn_output[:, t+1, :], self.time, method=self.integrator)[-1]
        return ode_output
