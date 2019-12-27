# Lint as: python3
"""Pseudocode description of the MuZero algorithm."""
# pylint: disable=unused-argument
# pylint: disable=missing-docstring
# pylint: disable=g-explicit-length-test

import collections
import math
import typing
from typing import Dict, List, Optional

import numpy

##########################
####### Helpers ##########

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value


class MuZeroConfig(object):

  def __init__(self,
               action_space_size: int,
               max_moves: int,
               discount: float,
               dirichlet_alpha: float,
               num_simulations: int,
               batch_size: int,
               td_steps: int,
               num_actors: int,
               lr_init: float,
               lr_decay_steps: float,
               visit_softmax_temperature_fn,
               known_bounds: Optional[KnownBounds] = None):
    ### Self-Play
    self.action_space_size = action_space_size
    self.num_actors = num_actors

    self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
    self.max_moves = max_moves
    self.num_simulations = num_simulations
    self.discount = discount

    # Root prior exploration noise.
    self.root_dirichlet_alpha = dirichlet_alpha
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
    self.known_bounds = known_bounds

    ### Training
    self.training_steps = int(1000e3)
    self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = batch_size
    self.num_unroll_steps = 5
    self.td_steps = td_steps

    self.weight_decay = 1e-4
    self.momentum = 0.9

    # Exponential learning rate schedule
    self.lr_init = lr_init
    self.lr_decay_rate = 0.1
    self.lr_decay_steps = lr_decay_steps

  def new_game(self):
    return Game(self.action_space_size, self.discount)


def make_board_game_config(action_space_size: int, max_moves: int,
                           dirichlet_alpha: float,
                           lr_init: float) -> MuZeroConfig:

  def visit_softmax_temperature(num_moves, training_steps):
    if num_moves < 30:
      return 1.0
    else:
      return 0.0  # Play according to the max.

  return MuZeroConfig(
      action_space_size=action_space_size,
      max_moves=max_moves,
      discount=1.0,
      dirichlet_alpha=dirichlet_alpha,
      num_simulations=800,
      batch_size=2048,
      td_steps=max_moves,  # Always use Monte Carlo return.
      num_actors=1,
      lr_init=lr_init,
      lr_decay_steps=400e3,
      visit_softmax_temperature_fn=visit_softmax_temperature,
      known_bounds=KnownBounds(-1, 1))


def make_go_config() -> MuZeroConfig:
  return make_board_game_config(
      action_space_size=362, max_moves=722, dirichlet_alpha=0.03, lr_init=0.01)


def make_chess_config() -> MuZeroConfig:
  return make_board_game_config(
      action_space_size=4672, max_moves=512, dirichlet_alpha=0.3, lr_init=0.1)


def make_shogi_config() -> MuZeroConfig:
  return make_board_game_config(
      action_space_size=11259, max_moves=512, dirichlet_alpha=0.15, lr_init=0.1)


def make_tictactoe_config() -> MuZeroConfig:
  return make_board_game_config(
    action_space_size=9, max_moves=9, dirichlet_alpha=0.3, lr_init=0.1
  )

def make_atari_config() -> MuZeroConfig:

  def visit_softmax_temperature(num_moves, training_steps):
    if training_steps < 500e3:
      return 1.0
    elif training_steps < 750e3:
      return 0.5
    else:
      return 0.25

  return MuZeroConfig(
      action_space_size=18,
      max_moves=27000,  # Half an hour at action repeat 4.
      discount=0.997,
      dirichlet_alpha=0.25,
      num_simulations=50,
      batch_size=1024,
      td_steps=10,
      num_actors=350,
      lr_init=0.05,
      lr_decay_steps=350e3,
      visit_softmax_temperature_fn=visit_softmax_temperature)


class Action(object):

  def __init__(self, index: int):
    self.index = index

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    return self.index == other.index

  def __gt__(self, other):
    return self.index > other.index


class Player(object):

    def __init__(self, mark):
        mark_code_map = {'O': 1, 'X': 2}
        self.code = mark_code_map[mark]


class Node(object):

  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.hidden_state = None
    self.reward = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count


class ActionHistory(object):
    """Simple history container used inside the search.

    Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        if len(self.history) % 2 == 0:
            return Player('O')
        else:
            return Player('X')


class Environment(object):
    """The environment MuZero is interacting with."""

    def step(self, action):
        pass


import numpy as np
from gym_tictactoe.env import TicTacToeEnv


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, action_space_size: int, discount: float):
        self.environment = TicTacToeEnv()  # Game specific environment.
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount

    def terminal(self) -> bool:
    # Game specific termination rules.
        return self.environment.done

    def legal_actions(self) -> List[Action]:
    # Game specific calculation of legal actions.
        return [Action(pos) for pos in range(len(self.environment.board)) if self.environment.board[pos] == 0]

    def apply(self, action: Action):
        _, reward, _, _ = self.environment.step(action.index)
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a.index].visit_count / sum_visits if a.index in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_image(self, state_index: int):
    # Game specific feature planes.
        return np.array(self.environment.board).reshape(1, 1, 1, -1)

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int,
                  to_play: Player):
    # The value target is the discounted root value of the search tree N steps
    # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
          bootstrap_index = current_index + td_steps
          if bootstrap_index < len(self.root_values):
            value = self.root_values[bootstrap_index] * self.discount**td_steps
          else:
            value = 0

          for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
            value += reward * self.discount**i  # pytype: disable=unsupported-operands

          if current_index < len(self.root_values):
            targets.append((value, self.rewards[current_index],
                            self.child_visits[current_index]))
          else:
            # States past the end of games are treated as absorbing states.
            targets.append((0, 0, []))
        return targets

    def to_play(self) -> Player:
        return Player(self.environment.mark)

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)


class ReplayBuffer(object):

  def __init__(self, config: MuZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self, num_unroll_steps: int, td_steps: int):
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    return [(g.make_image(i), g.history[i:i + num_unroll_steps],
             g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
            for (g, i) in game_pos]

  def sample_game(self) -> Game:
    # Sample game from buffer either uniformly or according to some priority.
    return np.random.choice(self.buffer)

  def sample_position(self, game) -> int:
    # Sample position from game either uniformly or according to some priority.
    return np.random.choice(game.history)


import torch
from torch import nn
from tic_tac_toe_models import Representation, Prediction, Dynamics


class NetworkOutput(typing.NamedTuple):
  value: torch.Tensor
  reward: torch.Tensor
  policy_logits: Dict[Action, float]
  hidden_state: List[float]


class Network(nn.Module):
    def __init__(self, num_blocks, channels_in, size_x, size_y, latent_dim, action_space_size):
        super().__init__()
        self.steps = 0
        self.representation = Representation(num_blocks, channels_in, latent_dim)
        self.prediction = Prediction(num_blocks, latent_dim, size_x, size_y, action_space_size, latent_dim)
        self.dynamics = Dynamics(num_blocks, size_x, size_y,
                                 state_channels_in=latent_dim,
                                 action_channels_in=1,
                                 latent_dim=latent_dim)

        self.eval()

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        image = torch.Tensor(image)
        initial_state = self.representation(image)
        policy_logits, value = self.prediction(initial_state)
        return NetworkOutput(value, torch.Tensor([0]), policy_logits, initial_state)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        action_arr = np.zeros(config.action_space_size)
        action_arr[action] = 1
        action_arr = action_arr.reshape(1, 1, 1, -1)

        # dynamics + prediction function
        hidden_state = torch.Tensor(hidden_state)
        action_tensor = torch.Tensor(action_arr)

        next_state, reward = self.dynamics((hidden_state, action_tensor))
        policy_logits, value = self.prediction(next_state)

        return NetworkOutput(value, reward, policy_logits, next_state)

    def get_weights(self):
        # Returns the weights of this network.
        return []

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.steps


class SharedStorage(object):

  def __init__(self):
    self._networks = {}

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.keys())]
    else:
      # policy -> uniform, value -> 0, reward -> 0
      return make_uniform_network()

  def save_network(self, step: int, network: Network):
    self._networks[step] = network


##### End Helpers ########
##########################


import threading
import time


# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    threads = []
    print('Launching {} self play job(s)...'.format(config.num_actors))
    for _ in range(config.num_actors):
        t = threading.Thread(target=launch_job, args=(run_selfplay, config, storage, replay_buffer), daemon=True)
        threads.append(t)
        # launch_job(run_selfplay, config, storage, replay_buffer)

    for thread in threads:
        thread.start()

    while len(replay_buffer.buffer) == 0:
        time.sleep(5)

    print('Training network...')
    train_network(config, storage, replay_buffer)

    return storage.latest_network()


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: MuZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
    game_counter = 0
    while True:
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)

        game_counter += 1
        if game_counter % 100 or game_counter == 1:
            print('Worker so far played {} games...'.format(game_counter))


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network) -> Game:
    game = config.new_game()

    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(root, game.to_play(), game.legal_actions(),
                    network.initial_inference(current_observation))
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network)
        game.apply(Action(action))
        game.store_search_statistics(root)

    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory,
             network: Network):
  min_max_stats = MinMaxStats(config.known_bounds)

  for _ in range(config.num_simulations):
    history = action_history.clone()
    node = root
    search_path = [node]

    while node.expanded():
      action, node = select_child(config, node, min_max_stats)
      history.add_action(action)
      search_path.append(node)

    # Inside the search tree we use the dynamics function to obtain the next
    # hidden state given an action and the previous hidden state.
    parent = search_path[-2]
    network_output = network.recurrent_inference(parent.hidden_state,
                                                 history.last_action())
    expand_node(node, history.to_play(), history.action_space(), network_output)

    backpropagate(search_path, network_output.value, history.to_play(),
                  config.discount, min_max_stats)


def select_action(config: MuZeroConfig, num_moves: int, node: Node,
                  network: Network):
  visit_counts = [
      (child.visit_count, action) for action, child in node.children.items()
  ]
  t = config.visit_softmax_temperature_fn(
      num_moves=num_moves, training_steps=network.training_steps())
  _, action = softmax_sample(visit_counts, t)
  return action


# Select the child with the highest UCB score.
def select_child(config: MuZeroConfig, node: Node,
                 min_max_stats: MinMaxStats):
  _, action, child = max(
      (ucb_score(config, node, child, min_max_stats), action,
       child) for action, child in node.children.items())
  return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: MuZeroConfig, parent: Node, child: Node,
              min_max_stats: MinMaxStats) -> float:
  pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  value_score = min_max_stats.normalize(child.value())
  return prior_score + value_score


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(node: Node, to_play: Player, actions: List[Action],
                network_output: NetworkOutput):
  node.to_play = to_play
  node.hidden_state = network_output.hidden_state
  node.reward = network_output.reward
  policy = {a: math.exp(network_output.policy_logits[a.index]) for a in actions}
  policy_sum = sum(policy.values())
  for action, p in policy.items():
    node.children[action.index] = Node(p / policy_sum)


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play: Player,
                  discount: float, min_max_stats: MinMaxStats):
  for node in search_path:
    node.value_sum += value if node.to_play == to_play else -value
    node.visit_count += 1
    min_max_stats.update(node.value())

    value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MuZeroConfig, node: Node):
  actions = list(node.children.keys())
  noise = numpy.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########

global_step = 0


def train_network(config: MuZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
    global global_step
    # network = Network()
    network = make_uniform_network()
    global_step += 1
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=config.lr_init,
                                weight_decay=config.lr_decay_rate,
                                momentum=config.momentum)

    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        update_weights(optimizer, network, batch, config.weight_decay)
    storage.save_network(config.training_steps, network)


def update_weights(optimizer: torch.optim.Optimizer, network: Network, batch, weight_decay: float):
    network.train()
    optimizer.zero_grad()

    mse_criterion = torch.nn.MSELoss()
    # cross_entropy_criterion = torch.nn.CrossEntropyLoss()
    bce_with_logits_criterion = torch.nn.BCEWithLogitsLoss()

    reward_loss, value_loss, policy_loss = 0, 0, 0
    for image, actions, targets in batch:
        # Initial step, from the real observation.
        value, reward, policy_logits, hidden_state = network.initial_inference(image)
        predictions = [(1.0, value, reward, policy_logits)]

        # Recurrent steps, from action and previous hidden state.
        for action in actions:
            value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action)
            predictions.append((1.0 / len(actions), value, reward, policy_logits))

            #TODO: no need to scale gradients in half?

        for prediction, target in zip(predictions, targets):
            gradient_scale, value, reward, policy_logits = prediction
            target_value, target_reward, target_policy = target

            reward_loss += mse_criterion(torch.Tensor([target_reward]), reward)
            value_loss += mse_criterion(torch.Tensor([target_value]), value)

            #TODO: need to transform policy or target policy from/to int/array?
            # and/or target policy to tensor?
            policy_loss += bce_with_logits_criterion(torch.Tensor(target_policy), policy_logits)

    total_loss = reward_loss + value_loss + policy_loss

    total_loss.backward()
    optimizer.step()
    network.steps += 1

    # reporting
    avg_policy_loss = policy_loss.item() / len(batch)
    avg_reward_loss = reward_loss.item() / len(batch)
    avg_value_loss = value_loss.item() / len(batch)
    avg_total_loss = total_loss.item() / len(batch)
    print('total_loss: {} | policy loss: {} | reward loss: {} | value loss: {}'
          .format(avg_total_loss, avg_policy_loss, avg_reward_loss, avg_value_loss))


def scalar_loss(prediction, target) -> float:
  # MSE in board games, cross entropy between categorical values in Atari.
  return -1

######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################


# Stubs to make the typechecker happy.
def softmax_sample(distribution, temperature: float):
    # from: https://github.com/Zeta36/muzero/blob/master/muzero.py
    if temperature == 0:
        temperature = 1

    distribution = numpy.array(distribution) ** (1 / temperature)
    p_sum = distribution.sum()
    sample_temp = distribution / p_sum
    return 0, numpy.argmax(numpy.random.multinomial(1, sample_temp, 1))


def launch_job(f, *args):
    f(*args)


def make_uniform_network():
    return Network(num_blocks=2, channels_in=1, size_x=9, size_y=1, latent_dim=4, action_space_size=9)


config = make_tictactoe_config()
# vs_random_once = random_vs_random()
# print('random_vs_random = ', sorted(vs_random_once.items()), end='\n')
network = muzero(config)