import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive


class DMM(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """

    def __init__(self, emitter, transition, combiner, encoder,
                 z_dim, encoder_shape,
                 num_iafs=0, iaf_dim=50, use_cuda=False):
        super().__init__()
        # instantiate pytorch modules used in the model and guide below
        self.emitter = emitter
        self.trans = transition
        self.combiner = combiner
        self.encoder = encoder

        # if we're using normalizing flows, instantiate those too
        self.iafs = [affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)

        # define a (trainable) parameters z_0 and z_q_0 that help define
        # the probability distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(encoder_shape))

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all pytorch (sub)modules
        if use_cuda:
            self.cuda()

    # the model p(x_{1:T} | z_{1:T}) p(z_{1:T})
    def model(self, mini_batch, mini_batch_mask, annealing_factor=1.0):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))

        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("z_minibatch", len(mini_batch)):
            # sample the latents z and observed x's one time step at a time
            # we wrap this loop in pyro.markov so that TraceEnum_ELBO can use multiple samples from the guide at each z
            for t in pyro.markov(range(1, T_max + 1)):
                # the next chunk of code samples z_t ~ p(z_t | z_{t-1})
                # note that (both here and elsewhere) we use poutine.scale to take care
                # of KL annealing. we use the mask() method to deal with raggedness
                # in the observed data (i.e. different sequences in the mini-batch
                # have different lengths)

                # first compute the parameters of the diagonal gaussian distribution p(z_t | z_{t-1})
                z_loc, z_scale = self.trans(z_prev)

                # then sample z_t according to dist.Normal(z_loc, z_scale)
                # note that we use the reshape method so that the univariate Normal distribution
                # is treated as a multivariate Normal distribution with a diagonal covariance.
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t,
                                      dist.Normal(z_loc, z_scale)
                                      .mask(mini_batch_mask[:, t - 1:t])
                                      .to_event(1))

                # compute the probabilities that parameterize the normal likelihood
                emission_loc_t, emission_scale_t = self.emitter(z_t)
                # the next statement instructs pyro to observe x_t according to the
                # bernoulli distribution p(x_t|z_t)
                pyro.sample("obs_x_%d" % t,
                            dist.Normal(emission_loc_t, emission_scale_t)
                            .mask(mini_batch_mask[:, t - 1:t])
                            .to_event(1),
                            obs=mini_batch[:, t - 1, :])
                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev = z_t

    # the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
    def guide(self, mini_batch, mini_batch_mask, annealing_factor=1.0):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)

        # push the observed x's through the rnn;
        # encoder_output contains the hidden state at each time step
        encoder_output = self.encoder(mini_batch, T_max)

        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))

        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(mini_batch)):
            # sample the latents z one time step at a time
            # we wrap this loop in pyro.markov so that TraceEnum_ELBO can use multiple samples from the guide at each z
            for t in pyro.markov(range(1, T_max + 1)):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
                z_loc, z_scale = self.combiner(z_prev, encoder_output[:, t - 1, :])

                # if we are using normalizing flows, we apply the sequence of transformations
                # parameterized by self.iafs to the base distribution defined in the previous line
                # to yield a transformed distribution that we use for q(z_t|...)
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                    assert z_dist.event_shape == (self.z_q_0.size(0),)
                    assert z_dist.batch_shape[-1:] == (len(mini_batch),)
                else:
                    z_dist = dist.Normal(z_loc, z_scale)
                    assert z_dist.event_shape == ()
                    assert z_dist.batch_shape[-2:] == (len(mini_batch), self.z_q_0.size(0))

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=annealing_factor):
                    if len(self.iafs) > 0:
                        # in output of normalizing flow, all dimensions are correlated (event shape is not empty)
                        z_t = pyro.sample("z_%d" % t,
                                          z_dist.mask(mini_batch_mask[:, t - 1]))
                    else:
                        # when no normalizing flow used, ".to_event(1)" indicates latent dimensions are independent
                        z_t = pyro.sample("z_%d" % t,
                                          z_dist.mask(mini_batch_mask[:, t - 1:t])
                                          .to_event(1))
                # the latent sampled at this time step will be conditioned upon in the next time step
                # so keep track of it
                z_prev = z_t

    # define a helper function for reconstructing images
    def reconstruction(self, x):
        # infer all x_t from ODE-RNN
        T_max = x.size(1)

        encoder_output = self.encoder(x, T_max)
        z_prev = self.z_q_0.data
        # step by step trans in the other direction

        Z = torch.zeros([1, T_max-1, self.trans.z_dim])
        Z_gen = torch.zeros([1, T_max-1, self.trans.z_dim])
        Z_gen_scale = torch.zeros([1, T_max-1, self.trans.z_dim])
        Obs = torch.zeros([1, T_max-1, self.emitter.input_dim])
        Obs_scale = torch.zeros([1, T_max-1, self.emitter.input_dim])

        for t in range(1, T_max):
            z_loc, z_scale = self.combiner(z_prev, encoder_output[:, t - 1, :])
            z_gen_loc, z_gen_scale = self.trans(z_loc)
            obs_loc, obs_scale = self.emitter(z_gen_loc)
            z_prev = z_loc
            Z[:, t - 1] = z_loc[:, 0]
            Z_gen[:, t - 1] = z_gen_loc[:, 0]
            Obs[:, t - 1] = obs_loc[:, 0]
            Obs_scale[:, t - 1] = obs_scale[:, 0]
            Z_gen_scale[:, t - 1] = z_gen_scale[:, 0]

        return Z, Z_gen, Z_gen_scale, Obs, Obs_scale
