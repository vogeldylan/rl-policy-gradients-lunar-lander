import numpy as np
import matplotlib.pyplot as plt

import time

import scipy.signal
from gym.spaces import Box, Discrete

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.distributions.categorical import Categorical

PHI_TYPES = ['reward-to-go', 'generalized-advantage-estimate']

def discount_cumsum(x, discount):
    """
    Compute  cumulative sums of vectors.

    Input: [x0, x1, ..., xn]
    Output: [x0 + discount * x1 + discount^2 * x2 + ... , x1 + discount * x2 + ... , ... , xn]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def combined_shape(length, shape=None):
    """Helper function that combines two array shapes."""
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    """The basic multilayer perceptron architecture used."""
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPCategoricalActor(nn.Module):
    """A class for the policy network."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        """Takes the observation and outputs a distribution over actions."""
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        """
        Takes a distribution and action, then gives the log-probability of the action
        under that distribution.
        """
        return pi.log_prob(act)

    def forward(self, obs, act=None):
        """
        Produce action distributions for given observations, and then compute the
        log-likelihood of given actions under those distributions.
        """
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCritic(nn.Module):
    """The network used by the value function."""
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Critical to ensure v has right shape
        return torch.squeeze(self.v_net(obs), -1)



class MLPActorCritic(nn.Module):
    """Class to combine policy (actor) and value (critic) function neural networks."""

    def __init__(self,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = 8

        # Build policy for 4-dimensional action space
        self.pi = MLPCategoricalActor(obs_dim, 4, hidden_sizes, activation)

        # Build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, state):
        """
        Take a state and return an action, value function, and log-likelihood
        of chosen action.
        """

        with torch.no_grad():
            # sample an action
            pi = self.pi._distribution(state)
            act = pi.sample()

            # sample the value function
            value = self.v.forward(state)

            # get the log-prob of the action under the distribution
            logp_act = self.pi._log_prob_from_distribution(pi, act)

        return int(act), float(value), float(logp_act)


class VPGBuffer:
    """
    Buffer to store trajectories.
    """
    def __init__(self, obs_dim, act_dim, size, gamma, lam):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        # advantage estimates
        self.phi_buf = np.zeros(size, dtype=np.float32)
        # rewards
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # trajectory's remaining return
        self.ret_buf = np.zeros(size, dtype=np.float32)
        # values predicted
        self.val_buf = np.zeros(size, dtype=np.float32)
        # log probabilities of chosen actions under behavior policy
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append a single timestep to the buffer. This is called at each environment
        update to store the outcome observed.
        """
        # buffer has to have room so you can store
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def end_traj(self, last_val=0):
        """
        Call after a trajectory ends. Last value is value(state) if cut-off at a
        certain state, or 0 if trajectory ended uninterrupted
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # compute temporal difference residuals of the value function
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[0:-1]
        # compute phi as per Eq. 16 of Schulman et al. (2018)
        self.phi_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # calculated the discounted reward-to-go
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[0:-1]

        self.path_start_idx = self.ptr


    def get(self):
        """
        Call after an epoch ends. Resets pointers and returns the buffer contents.
        """
        # Buffer has to be full before you can get something from it.
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        # normalize the phi values to zero mean and unit variance
        self.phi_buf = (self.phi_buf - self.phi_buf.mean()) / self.phi_buf.std()

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    phi=self.phi_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class Agent:
    def __init__(self, env):
        self.env = env
        self.hid = 64  # layer width of networks
        self.l = 2  # layer number of networks
        # initialises an actor critic
        self.ac = MLPActorCritic(hidden_sizes=[self.hid]*self.l)

        # use generalized advantage estimate for the policy gradient update weighting
        self.phi_type = PHI_TYPES[1]
        print(f'Using phi method: {self.phi_type}')

        # Learning rates for policy and value function
        pi_lr = 3e-3
        vf_lr = 1e-3

        # we will use the Adam optimizer to update value function and policy
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.v_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

    def pi_update(self, data):
        """
        Use the data from the buffer to update the policy. Returns nothing.
        """
        obs = data['obs']
        act = data['act']
        phi = data['phi']
        ret = data['ret']

        # reset gradients
        self.pi_optimizer.zero_grad()

        # get the log-prob of the policy for this episode
        logp_pi = self.ac.pi.forward(obs, act)[1]

        if self.phi_type == 'reward-to-go':
            # compute the loss by multiplying with the reward-to-go of each timestep
            loss = - (logp_pi * ret).sum()
        elif self.phi_type == 'generalized-advantage-estimate':
            # compute loss by multiplying with generalized advantage estimate of each timestep
            loss = - (logp_pi * phi).sum()

        # backwards step
        loss.backward()
        self.pi_optimizer.step()

        return

    def v_update(self, data):
        """
        Use the data from the buffer to update the value function. Returns nothing.
        """

        # number of value function updates to perform per epoch
        N_UPDATES = 100

        obs = data['obs']
        act = data['act']
        phi = data['phi']
        ret = data['ret']

        # size of each batch of data
        slice_size = int(len(obs) / N_UPDATES)

        # warn if data is lost
        if len(obs) % N_UPDATES:
            print(f'Warning: {len(obs)} were given, which does not cleanly divide into {N_UPDATES} updates. Some will be lost')

        for ind in range(N_UPDATES):
            self.v_optimizer.zero_grad()

            # create a new slice for this set of data
            data_slice = slice(ind*slice_size, (ind+1)*slice_size)

            # calculate the loss function
            # Note: here we know the exact future discounted return obtained for this slice,
            # so we can use that to create the squared error term
            loss = torch.square(self.ac.v.forward(obs[data_slice]) - ret[data_slice]).sum()

            # backwards pass
            loss.backward()
            self.v_optimizer.step()

        return

    def train(self):
        """
        Main training loop.

        IMPORTANT: This function is called by the checker to train your agent.
        You SHOULD NOT change the arguments this function takes and what it outputs!
        """

        # The observations are 8 dimensional vectors, and the actions are numbers,
        # i.e. 0-dimensional vectors (hence act_dim is an empty list).
        obs_dim = [8]
        act_dim = []

        # Training parameters
        # You may wish to change the following settings for the buffer and training
        # Number of training steps per epoch
        steps_per_epoch = 3000
        # Number of epochs to train for
        epochs = 50
        # The longest an episode can go on before cutting it off
        max_ep_len = 300
        # Discount factor for weighting future rewards
        gamma = 0.99
        lam = 0.97

        # Set up buffer
        buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

        # Initialize the ADAM optimizer using the parameters
        # of the policy and then value networks

        # Initialize the environment
        state, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main training loop: collect experience in env and update / log each epoch
        for epoch in range(epochs):
            ep_returns = []
            for t in range(steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(state, dtype=torch.float32))

                next_state, r, terminal = self.env.transition(a)
                ep_ret += r
                ep_len += 1

                # Log transition
                buf.store(state, a, r, v, logp)

                # Update state (critical!)
                state = next_state

                timeout = ep_len == max_ep_len
                epoch_ended = (t == steps_per_epoch - 1)

                if terminal or timeout or epoch_ended:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(state, dtype=torch.float32))
                    else:
                        v = 0
                    if timeout or terminal:
                        ep_returns.append(ep_ret)  # only store return when episode ended
                    buf.end_traj(v)
                    state, ep_ret, ep_len = self.env.reset(), 0, 0

            mean_return = np.mean(ep_returns) if len(ep_returns) > 0 else np.nan
            if len(ep_returns) == 0:
                print(f"Epoch: {epoch+1}/{epochs}, all episodes exceeded max_ep_len")
            print(f"Epoch: {epoch+1}/{epochs}, mean return {mean_return}")

            # This is the end of an epoch, so here is where we update the policy and value function

            data = buf.get()

            self.pi_update(data)
            self.v_update(data)


        return True


    def get_action(self, obs):
        """
        Sample an action from your policy.

        IMPORTANT: This function is called by the checker to evaluate your agent.
        You SHOULD NOT change the arguments this function takes and what it outputs!
        It is not used in your own training code. Instead the .step function in 
        MLPActorCritic is used since it also outputs relevant side-information. 
        """
        # return the action which maximizes the probability under the policy
        pi = self.ac.pi._distribution(torch.as_tensor(obs, dtype=torch.float32))
        return(int(pi.probs.argmax()))


def main():
    """
    Train and evaluate agent.

    This function basically does the same as the checker that evaluates your agent.
    You can use it for debugging your agent and visualizing what it does.
    """
    from lunar_lander import LunarLander
    from gym.wrappers.monitoring.video_recorder import VideoRecorder

    env = LunarLander()

    agent = Agent(env)
    agent.train()

    rec = VideoRecorder(env, "policy.mp4")
    episode_length = 300
    n_eval = 100
    returns = []
    print("Evaluating agent...")

    for i in range(n_eval):
        print(f"Testing policy: episode {i+1}/{n_eval}")
        state = env.reset()
        cumulative_return = 0
        # The environment will set terminal to True if an episode is done.
        terminal = False
        env.reset()
        for t in range(episode_length):
            if i <= 10:
                rec.capture_frame()
            # Taking an action in the environment
            action = agent.get_action(state)
            state, reward, terminal = env.transition(action)
            cumulative_return += reward
            if terminal:
                break
        returns.append(cumulative_return)
        print(f"Achieved {cumulative_return:.2f} return.")
        if i == 10:
            rec.close()
            print("Saved video of 10 episodes to 'policy.mp4'.")
    env.close()
    print(f"Average return: {np.mean(returns):.2f}")

if __name__ == "__main__":
    main()
