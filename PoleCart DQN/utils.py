import random
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from time import time

import gym
from gym.core import Env
from collections import deque

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim


class ReplayBuffer():
    def __init__(self, size:int, p_shuffle=0.2):
        """Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
            p_shuffle: probability to shuffle buffer appending a new item
        """
        self.size = size
        self.p = p_shuffle
        self.buffer = deque([], size)

    def push(self, transition)->list:
        """Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type

        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """
        self.buffer.append(transition)

        # shuffle a buffer to get rid of situation where close states are nearby
        if np.random.uniform() <= self.p:
          np.random.shuffle(self.buffer)

        return self.buffer

    def sample(self, batch_size:int)->list:
        """Get a random sample from the replay buffer

        Args:
            batch_size: size of sample

        Returns:
            iterable (e.g. list) with objects sampled from buffer without replacement
        """
        return random.sample(self.buffer, batch_size)

class DQN(nn.Module):
    def __init__(self, layer_sizes: list[int], activation_functions):
        """
        DQN initialisation

        Args:
            input_layer_size: number of input units
            layer_sizes: list with size of each layer as elements
            activation_functions: list of actiovation function for each layer
        """
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)])

        # we need some non-linearity
        self.functions = activation_functions

    def forward (self, x:torch.Tensor)->torch.Tensor:
        """Forward pass through the DQN

        Args:
            x: input to the DQN

        Returns:
            outputted value by the DQN
        """
        for layer, activation in zip(self.layers, self.functions):
          x = activation(layer(x))
        return x

def greedy_action(dqn:DQN, state:torch.Tensor)->int:
    """Select action according to a given DQN

    Args:
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Greedy action according to DQN
    """
    return int(torch.argmax(dqn(state)))

def epsilon_greedy(epsilon:float, dqn:DQN, state:torch.Tensor)->int:
    """Sample an epsilon-greedy action according to a given DQN

    Args:
        epsilon: parameter for epsilon-greedy action selection
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Sampled epsilon-greedy action
    """
    q_values = dqn(state)
    num_actions = q_values.shape[0]

    # old method commented
    """
    greedy_act = int(torch.argmax(q_values))
    p = float(torch.rand(1))
    if p > epsilon:
      return greedy_act
    else:
      return random.randint(0,num_actions-1)
    """

    # go to numpy
    q_values = q_values.cpu().detach().numpy()

    # get best actions ==> mask has True on the best actions
    mask = abs(q_values - np.max(q_values)) <= 1e-7

    probs = np.zeros(mask.shape)
    probs[mask] = (1 - epsilon) / np.sum(mask)
    probs += epsilon / num_actions

    # return action according to e-greedy policy
    return np.random.choice(np.arange(num_actions), p=probs)

def update_target(target_dqn:DQN, policy_dqn:DQN):
    """Update target network parameters using policy network.
    Does not return anything but modifies the target network passed as parameter

    Args:
        target_dqn: target network to be modified in-place
        policy_dqn: the DQN that selects the action
    """

    target_dqn.load_state_dict(policy_dqn.state_dict())

def loss(policy_dqn:DQN, target_dqn:DQN,
         states:torch.Tensor, actions:torch.Tensor,
         rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor, gamma:torch.Tensor)->torch.Tensor:
    """Calculate Bellman error loss

    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
        states: batched state tensor
        actions: batched action tensor
        rewards: batched rewards tensor
        next_states: batched next states tensor
        dones: batched Boolean tensor, True when episode terminates
        gamma: discount factor
    Returns:
        Float scalar tensor with loss value
    """

    bellman_targets = gamma * (~dones).reshape(-1) * (target_dqn(next_states)).max(1).values + rewards.reshape(-1)
    q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    return ((q_values - bellman_targets)**2).mean()

## Calculate number of episodes for base-value
def get_n_episodes(base):

  return math.ceil(math.log(0.05, base) / 100) * 100

## E-greedy sequence generator
def eps_sequence(delay, base=0.995):
  eps = 1.0
  k = 1

  while True:
    yield eps
    eps = eps * base if k >= delay else 1.0
    k += 1
 

## Test the agent
def test_agent(model, env, n_runs=50):
  """
  This function tests the learned agent

  Args:
        model: Model to be tested
        env: Environment
        n_runs: Number of episodes
    Returns:
        episodes_duration: Array that contains a duration for each episode
  """

  model.eval()
  episode_durations = []

  for run in range(n_runs):
    observation = env.reset()
    state = torch.tensor(observation).float()

    done = False
    terminated = False
    t = 0
    while not (done or terminated):

        # Select an action
        action = greedy_action(model, state)

        # Perform an action
        observation, reward, done, info = env.step(action)
        reward = torch.tensor([reward])
        action = torch.tensor([action])
        next_state = torch.tensor(observation).reshape(-1).float()

        # Move to the next state
        state = next_state

        t += 1

        if done or terminated:
              episode_durations.append(t)

  return np.array(episode_durations)
  
  
 # Train agent
def fit_agent(env, base, delay, buffer_size, p_shuffle, batch_size, collecting_steps, update_frequency, gamma, layers, activations, lr):
  """
  This function perfoms learning process of the agent

  Args:
        env: Environment to learn
        base: Exploration rate [0, 1]. The larger the value, the longer exploration phase
        delay: Number of episodes only for collecting
        buffer_size: Size of memory buffer
        p_shuffle: Probability to shuffle memory buffer appending new data
        batch_size: Size of batch through learning
        collecing_steps: Number of steps to collect data before learning
        update_frequency: How often policy_net updates target_net
        gamma: Discount factor for learning
        layers: list of size for each layer in the network
        actionvations: list of activation functions
        lr: learning rate for network
    Returns:
        agent: Target network. Agent that was learned
        buffer: The buffer after learning. I used it to check the distribution over states after learning. That was the most significant decision :)
        episode_durations: The reward for each episode
        loss_progress: The values of loss-function
        results_of_test: The results og the learned model
  """

  # timing
  start_time = time()

  # initialize network
  policy_net = DQN(layers, activations)
  target_net = DQN(layers, activations)

  update_target(target_net, policy_net)
  target_net.eval()

  # initialize optimizer
  #optimizer = optim.SGD(policy_net.parameters(), lr=lr)
  optimizer = optim.Adam(policy_net.parameters(), lr=lr)

  # initialize memory buffer
  memory = ReplayBuffer(buffer_size, p_shuffle)

  # some data for evaluation
  steps_done = 0
  episode_durations = []
  loss_progress = []

  # eps-generator for exploration
  eps_seq = eps_sequence(delay, base)
  N_EPISODES = get_n_episodes(base)

  print('Progress: ', end='')
  # firstly agent collects data and then starts using of e-greedy sequence
  ALL_EPISODES = delay + N_EPISODES
  for i_episode in range(ALL_EPISODES):

      eps = next(eps_seq)

      # start the environment
      observation = env.reset()
      state = torch.tensor(observation).float()

      done = False
      terminated = False
      t = 0
      loss_value = 0.0
      while not (done or terminated):

          # Select an action
          action = epsilon_greedy(eps, policy_net, state)

          # Perform an action
          observation, reward, done, info = env.step(action)
          reward = torch.tensor([reward])
          action = torch.tensor([action])
          next_state = torch.tensor(observation).reshape(-1).float()

          # It helps to collect more data but prevent massive spamming the buffer
          if np.random.uniform() <= eps or len(memory.buffer) <= 2 * collecting_steps:
            memory.push([state, action, next_state, reward, torch.tensor([done])])

          # Move to the next state
          state = next_state

          # Perform one step of the optimization (on the policy network)
          if len(memory.buffer) >= collecting_steps:
              transitions = memory.sample(batch_size)
              state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
              # Compute loss
              mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones, torch.tensor([gamma]))
              # Optimize the model
              optimizer.zero_grad()
              mse_loss.backward()
              optimizer.step()

              # save loss dynamic
              loss_value += mse_loss.item()

          if done or terminated:
              episode_durations.append(t + 1)
              loss_progress.append(loss_value / (t + 1))
          t += 1

      # Update the target network, copying all weights and biases in DQN
      if i_episode % update_frequency == 0:
          update_target(target_net, policy_net)

      if i_episode % (ALL_EPISODES // 10) == 0:
              print('{}% '.format(round(i_episode * 100 / ALL_EPISODES)), end='')

  print('100%')
  print('Learning time = {} sec.'.format(round(time() - start_time, 2)))

  # last update after learning
  update_target(target_net, policy_net)

  # test the model
  results_of_test = test_agent(target_net, env, n_runs=200)

  return target_net, memory.buffer, episode_durations, loss_progress, results_of_test
  
def plot_data_for_states(data, title, discrete=False):

  actions = ['push left', 'push right']
  
  # create subplots
  fig, axs = plt.subplots(1, len(list(data.keys())), figsize=(12, 4))

  fig.suptitle(title, size=12)

  # configure subplots with data
  for idx, key in enumerate(data):
    x_data, y_data, result = data[key]
    cs = axs[idx].contourf(x_data, y_data, result, alpha=0.4, cmap='cividis')
    axs[idx].set_title('Cart velocity = {}'.format(key), size=8)

  # make up
  for ax in axs.flat:
    ax.set(xlabel='Pole angle', ylabel='Pole velocity')
    ax.set_xlim(x_data.min(), x_data.max())
    ax.set_ylim(y_data.min(), y_data.max())
    ax.set_xticks(np.linspace(-0.2, 0.2, 5, endpoint=True))
    ax.set_yticks(np.linspace(y_data.min(), y_data.max(), 11, endpoint=True))
    ax.label_outer()

  # create a legend for subplot
  artists, labels = cs.legend_elements()
  if discrete:
    artists = [artists[0], artists[-1]]
    labels = actions
    box_param = (1.65, 1.02)
  else:
    box_param = (1.8, 1.02)

  plt.legend(handles=artists, labels=labels, bbox_to_anchor=box_param)

  plt.show()