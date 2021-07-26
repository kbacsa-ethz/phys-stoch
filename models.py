import torch
import torch.nn as nn
import utils
# from torch.autograd.functional import jacobian, hessian
from torch.autograd import grad

# from torchdiffeq.torchdiffeq import odeint
from torchdiffeq import odeint
    
class H_func(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20, hlayers=0):
        super(H_func, self).__init__()
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
    
class Grad_H_func(nn.Module):

    def __init__(self, z_dim=4, nhidden=20, hlayers=1):
        super(Grad_H_func, self).__init__()
        self.tanh = nn.Tanh()

        self.linears = nn.ModuleList([])
        self.linears.append(nn.Linear(z_dim, nhidden))

        for layer in range(hlayers):
            self.linears.append(nn.Linear(nhidden, nhidden))

        self.fc = nn.Linear(nhidden, z_dim)
        self.nlayers = len(self.linears)
        self.nfe = 0
        self.softplus = nn.Softplus()
        
        self.H_grad_net = nn.Sequential(
            nn.Linear(z_dim, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(),                
            nn.Linear(nhidden, z_dim),    
                    ) 

    def forward(self, t, x):
        # out = x.clone()
        # for layer in range(self.nlayers):
        #     out = self.tanh(self.linears[layer](out))
        # out = self.fc(out)   
        
        out = self.H_grad_net(x)
        
        return out

class lagrangian_ode_func(nn.Module):
    def __init__(self, z_dim = 2, hidden_dim = 15):
        super(lagrangian_ode_func, self).__init__()
        self.z_dim = z_dim
        self.n_dim = z_dim // 2
                  
        self.T_net = nn.Sequential(
                nn.Linear(self.n_dim, hidden_dim),
                nn.Softplus(),
                # nn.Linear(hidden_dim, hidden_dim),
                # nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus(),
            )
        
        self.V_net = nn.Sequential(
                nn.Linear(self.n_dim, hidden_dim),
                nn.Softplus(),
                # nn.Linear(hidden_dim, hidden_dim),
                # nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus(),
            ) 
        
        self.T_V_net = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),                
                nn.Linear(hidden_dim, 1),    
            )        
        self.m1 = 1.0
        self.m2 = 1.0
        self.k1 = nn.Parameter(torch.ones(1)) # ground-trurth value = 2.9
        self.k2 = nn.Parameter(torch.ones(1)) # ground-truth value = 0.6
        
    def forward(self, t, x):
        with torch.set_grad_enabled(True):
            self.n = n = x.shape[1]//2
            qqd = x.requires_grad_(True)
            q = qqd[:,:self.n]
            qd = qqd[:,self.n:]  
            
            
            # L = self.T_net(qd).sum() - self.V_net(q).sum() # Lagrangian 
            
            # L = self.T_V_net(qqd).sum()
            # J = grad(L, qqd, create_graph=True)[0] 
            # DL_q, DL_qd = J[:,:n], J[:,n:]

            
            # T = self.T_net(qd).sum() 
            
            T = 0.5*(qd**2).sum()
            # V = self.V_net(q).sum()
            V = self._potential_E(q)
            # # T, V = self._ground_truth_T_V(q, qd)
            # # T = T.sum()
            # # V = V.sum()
                        
            J1 = grad(T,qd, create_graph=True)[0]
            J2 = grad(-V,q, create_graph=True)[0]
            
            DL_q = J2
            DL_qd = J1            
                        
            DDL_qd = []
            for i in range(n):
                J_qd_i = DL_qd[:,i][:,None]
                H_i = grad(J_qd_i.sum(), qqd, create_graph=True)[0][:,:,None]
                DDL_qd.append(H_i)
            DDL_qd = torch.cat(DDL_qd, 2)
            
            DDL_qqd, DDL_qdqd = DDL_qd[:,:n,:], DDL_qd[:,n:,:]
            
            # DDL_qdqd_inverse, info = torch.linalg.inv_ex( DDL_qdqd)
            # idx = torch.nonzero(info)
            
            # DDL_qdqd[idx,:,:] = 1e-5*torch.normal(0,1,size = (n,n))
            # DDL_qdqd[idx,:,:] = 1e-10*torch.diag(torch.randn(n))
            # print(DDL_qdqd[idx,:,:])
            DDL_qdqd_inverse, _ = torch.linalg.inv_ex( DDL_qdqd)
            
            T = torch.einsum('ijk, ij -> ik', DDL_qqd, qqd[:,n:])
            # qdd = torch.einsum('ijk, ij -> ik', DDL_qdqd.inverse(), DL_q - T)
            # qdd = torch.einsum('ijk, ij -> ik', 
                                # torch.inverse( DDL_qdqd + 1e-8*torch.diag(torch.randn(n)) ), DL_q)
            qdd = torch.einsum('ijk, ij -> ik', 
                                DDL_qdqd_inverse , DL_q)            
        return torch.cat([qd, qdd], 1)
    def _lagrangian(self, qqt):
        
        # q, qt = torch.split(qqt, self.n_dim, dim = 1)  
        q = qqt[:,:self.n_dim]
        qt = qqt[:,self.n_dim:]
         
        # T, V = self._ground_truth_T_V(q,qt)
        
        T = self.T_net(qt)     # kinetic energy
        V = self.V_net(q)      # potential energy
        
        L = T - V               # lagrangian

        return L.sum() 
    
    def _ground_truth_T_V(self,q,qt):
         
        T = 0.5 * self.m1 * qt[:,0]**2 + 0.5*self.m2 * qt[:,1]**2 
        V = 0.5 * self.k1 * q[:,0]**2 + 0.5*self.k2 * (q[:,1] - q[:,0])**2
        
        return T, V
    def _potential_E(self,q):
        
        V = 0.5*self.k1*q[:,0]**2 + 0.5*self.k2*(q[:,0] - q[:,1])**2
        return V.sum()
    

class Emitter(nn.Module):
    """
    Parameterizes the observation likelihood p(x_t | z_t)
    """

    def __init__(self, input_dim, z_dim, emission_dim, h_layers):
        super().__init__()

        if h_layers == 0:
            self.hidden_to_loc = nn.Linear(z_dim, input_dim, bias= False)
            self.hidden_to_scale = nn.Linear(z_dim, input_dim, bias= False)
        else:
            self.hidden_to_loc = nn.Linear(emission_dim, input_dim, bias= True )
            self.hidden_to_scale = nn.Linear(emission_dim, input_dim, bias= True)

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
        self.Cd =  torch.tensor([
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1],
                                  [-3.5, 0.6, 0.0, 0,],
                                  [0.6, -0.6,0, 0.0 ]
                                  ])
        
        self.fixed_emitter = nn.Linear(z_dim,input_dim, bias = False)
        self.fixed_emitter.weight.data = self.Cd
        for param in self.fixed_emitter.parameters():
            param.requires_grad = False 
        

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
        # loc = self.fixed_emitter(z_t)
        scale = self.e_activation(self.hidden_to_scale(x))
        return loc, scale


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability p(z_t | z_{t-1})
    See section 5 in the reference for comparison.
    """

    def __init__(self, z_dim, transition_dim, dt):
        super().__init__()
        # initialize the six linear transformations used in the neural network
        self.A = torch.tensor([
                                [0,0,1,0],
                                [0,0,0,1],
                                [-5,4.3,0,0],
                                [4.3,-4.3,0,0]
                                    ])
        self.Ad = torch.matrix_exp(self.A * 0.02)
        
        self.fixed_trans = nn.Linear(z_dim,z_dim, bias = False)
        self.fixed_trans.weight.data = self.Ad
        for param in self.fixed_trans.parameters():
            param.requires_grad = False 
        
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        # self.lin_proposed_mean_z_to_z = PosSemiDefLayer(z_dim)
        # self.lin_proposed_mean_z_to_z = nn.Linear(z_dim,z_dim,bias = False)
        
        self.lin_gate_hidden_to_hidden = nn.Linear(transition_dim, transition_dim)
        self.lin_hidden_to_scale = nn.Linear(transition_dim,z_dim)
        
        self.lin_hidden_to_loc = nn.Linear(transition_dim, z_dim)
        # 
        # self.ode_func_tran = Grad_H_func(z_dim, 20, 1)
        self.ode_func_trans = lagrangian_ode_func(z_dim,20 )
        
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        # self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        # self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.z_dim = z_dim
        
        self.time = torch.arange(0, dt + dt, dt)

    def forward(self, z_t_1):
        """
        Given the latent z_{t-1} corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution p(z_t | z_{t-1})
        """
        # compute the gating function
        # _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        # gate = self.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # proposed_mean = self.lin_proposed_mean_z_to_z(z_t_1)
        # loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        

        h1 = self.relu( self.lin_gate_z_to_hidden(z_t_1) )
        h2 = self.relu( self.lin_gate_hidden_to_hidden(h1))
        
        # loc =  self.lin_hidden_to_loc(h2)
        loc = odeint(self.ode_func_trans, z_t_1, self.time, 
                        # method='yoshida4th',
                        # method = "dopri8"
                        )[-1]

        scale = self.softplus(self.lin_hidden_to_scale(h2))
        
        return loc, scale

class Combiner_new(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the encoder
    """

    def __init__(self, z_dim, encoder_dim,dt,device):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, encoder_dim)
        self.lin_hidden_to_loc = nn.Linear(encoder_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(encoder_dim, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        # self.ode_func_combiner = lagrangian_ode_func(z_dim, 15)
        self.ode_func_combiner = Grad_H_func(z_dim, 16)
        
        self.time = torch.arange(0,dt+dt,dt).to(device)
        
    def forward(self, z_t_1, h1, h2):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the encoder `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`
        """
        # combine the encoder hidden state with a transformed version of z_t_1
        h_combined = 1/3.0 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h1 + h2)
        # use the combined hidden state to compute the mean used to sample z_t
        # loc_ = self.lin_hidden_to_loc(h_combined)
        # loc = odeint(self.ode_func_combiner, loc_, self.time,
        #               # method = "dopri5"
        #               method = "yoshida4th",
        #              )[-1]
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
        self.num_layers = num_layers

    def forward(self, x, seq_len):
        x_reversed = utils.reverse_sequences(x, [seq_len] * x.size(0))

        # do sequence packing
        x_reversed = nn.utils.rnn.pack_padded_sequence(x_reversed,
                                                       [seq_len] * x.size(0),
                                                       batch_first=True)

        h_0_contig = self.h_0.expand(self.num_layers, x.size(0), self.rnn.hidden_size).contiguous()
        rnn_output, _ = self.rnn(x_reversed, h_0_contig)
        rnn_output = utils.pad_and_reverse(rnn_output, [seq_len] * x.size(0))
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

        self.fc = nn.Linear(nhidden, 1)
        self.nlayers = len(self.linears)
        self.nfe = 0
        self.softplus = nn.Softplus()

    def forward(self, t, x):
        out = x.clone()
        for layer in range(self.nlayers):
            out = self.tanh(self.linears[layer](out))
        out = self.softplus(self.fc(out))
        return out


class SymplecticODEEncoder(nn.Module):
    """
    Parameterizes `q(z_t | x_{t:T})`
    """

    def __init__(self, input_size, z_dim, hidden_dim, n_layers, non_linearity, batch_first, rnn_layers, dropout, dt, discretization):
        super().__init__()

        # self.latent_func = GradPotentialODEfunc(z_dim, hidden_dim, n_layers)
        # self.latent_func = PotentialODEfunc(z_dim, hidden_dim, n_layers)

        self.rnn = nn.RNN(input_size=input_size, hidden_size=z_dim, nonlinearity=non_linearity,
                          batch_first=batch_first, bidirectional=False, num_layers=rnn_layers, dropout=dropout)

        self.h_0 = nn.Parameter(torch.zeros(rnn_layers, 1, z_dim))
        self.rnn_layers = rnn_layers
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
            ode_output[:, t, :] = odeint(self.latent_func, rnn_output[:, t+1, :], self.time, method='yoshida4th')[-1]
        return ode_output

class LagrangianODEEncoder(nn.Module):
    """
    Parameterizes `q(z_t | x_{t:T})`
    """

    def __init__(self, input_size, z_dim, hidden_dim, n_layers, non_linearity, batch_first, rnn_layers, dropout, dt, discretization):
        super().__init__()

        self.latent_func = lagrangian_ode_func(z_dim, hidden_dim)

        self.rnn = nn.RNN(input_size=input_size, hidden_size= hidden_dim, nonlinearity=non_linearity,
                          batch_first=batch_first, bidirectional = True, 
                          num_layers=rnn_layers, dropout=dropout)

        self.h_0 = nn.Parameter(torch.zeros(rnn_layers, 1, hidden_dim))
        self.rnn_layers = rnn_layers
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
            ode_output[:, t, :] = odeint(self.latent_func, rnn_output[:, t+1, :], self.time)[-1]
        return ode_output

class BiRNNEncoder(nn.Module):
    """
    Parameterizes `q(z_t | x_{t:T})`
    """

    def __init__(self, input_size, hidden_size,  num_layers, dropout, seq_len):
        super().__init__()

        self.rnn = nn.GRU(input_size=input_size, hidden_size= hidden_size, 
                          batch_first=True, bidirectional=True, num_layers=num_layers,
                          dropout = dropout)
        self.h_0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, seq_len):

        h_0_contig = self.h_0.expand(2*self.num_layers, x.size(0), self.rnn.hidden_size).contiguous()
        rnn_output, _ = self.rnn(x, h_0_contig)
        rnn_output = rnn_output.view(rnn_output.size(0),rnn_output.size(1),
                                      2,self.rnn.hidden_size)
        h1 = rnn_output[:,:,0,:]
        h2 = rnn_output[:,:,1,:]        
        
        
        return h1, h2

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