import collections
import typing
from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn


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
                 init_env,
                 get_env_legal_actions,
                 get_env_obs,
                 get_to_play,
                 turn_based,
                 make_uniform_network,
                 known_bounds: Optional[KnownBounds]=None,
                 optimize_reward=True):
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
        self.train_report_interval = int(1e2)
        self.test_interval = int(1e3)
        self.window_size = int(1e4)
        self.batch_size = batch_size
        self.num_unroll_steps = 9
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

        # Added
        self.init_env = init_env
        self.optimize_reward = optimize_reward
        self.get_env_legal_actions = get_env_legal_actions
        self.get_env_obs = get_env_obs
        self.get_to_play = get_to_play
        self.make_uniform_network = make_uniform_network
        self.turn_based = turn_based

    def new_game(self):
        return Game(self.action_space_size, self.discount, self.init_env, self.get_env_legal_actions,
                    self.get_env_obs, self.get_to_play, self.turn_based)


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
        self.mark = mark
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

    def __init__(self, history: List[Action], action_space_size: int, turn_based=True):
        self.history = list(history)
        self.action_space_size = action_space_size
        self.turn_based = turn_based

    def clone(self):
        return ActionHistory(self.history, self.action_space_size, self.turn_based)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        if not self.turn_based or len(self.history) % 2 == 0:
            return Player('O')
        else:
            return Player('X')


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, action_space_size: int, discount: float,
                 init_env, get_env_legal_actions, get_env_obs, get_to_play, turn_based,
                 history=None,
                 rewards=None,
                 child_visits=None,
                 root_values=None,
                 done=False):
        self.init_env = init_env
        self.environment = init_env()  # Game specific environment.
        self.current_observation = self.environment.reset()
        self.history = history if history else []
        self.rewards = rewards if rewards else []
        self.child_visits = child_visits if child_visits else []
        self.root_values = root_values if root_values else []
        self.action_space_size = action_space_size
        self.discount = discount
        self.done = done
        self.get_env_legal_actions = get_env_legal_actions
        self.get_env_obs = get_env_obs
        self.get_to_play = get_to_play
        self.turn_based = turn_based

    def terminal(self) -> bool:
        # Game specific termination rules.
        return self.done

    def legal_actions(self) -> List[Action]:
        # Game specific calculation of legal actions.
        return [Action(legal_action) for legal_action in self.get_env_legal_actions(self.environment)]

    def apply(self, action: Action, save_history=True):
        obs, reward, done, _ = self.environment.step(action.index)
        self.current_observation = obs
        self.done = done

        if save_history:
            self.history.append(action)
            self.rewards.append(reward)

        return obs, reward, done

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a.index].visit_count / sum_visits if a.index in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_image(self, state_index=None):
        # Game specific feature planes.
        if state_index is not None:
            self.current_observation = self.environment.reset()

            for i in range(state_index):
                self.current_observation, _, _ = self.apply(self.history[i], save_history=False)

        return self.get_env_obs(self.current_observation)

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i  # pytype: disable=unsupported-operands

            if current_index < len(self.root_values):
                targets.append((value, self.rewards[current_index], self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, 0, []))
        return targets

    def to_play(self) -> Player:
        return Player(self.get_to_play(self.environment))

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size, self.turn_based)

    def clone(self):
        return Game(self.action_space_size,
                      self.discount, self.init_env,
                      self.get_env_legal_actions, self.get_env_obs,
                      self.get_to_play, self.turn_based,
                      list(self.history), list(self.rewards),
                      list(self.child_visits), list(self.root_values),
                      self.done)


class ReplayBuffer(object):

    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        buffer = list(self.buffer)
        if len(self.buffer) > self.window_size:
            buffer.pop(0)
        buffer.append(game)
        self.buffer = buffer

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i),
                 g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                for (g, i) in game_pos]

    def sample_game(self) -> Game:
        # Sample game from buffer either uniformly or according to some priority.
        return np.random.choice(self.buffer).clone()

    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return np.random.randint(len(game.history)) # 0: initial position, 1-action_space fast forward to history position


class NetworkOutput(typing.NamedTuple):
    value: torch.Tensor
    reward: torch.Tensor
    policy_logits: Dict[Action, float]
    hidden_state: List[float]


class Network(nn.Module):
    def __init__(self, modules, num_blocks, channels_in, size_x, size_y, latent_dim, action_space_size, device, config):
        super().__init__()
        self.channels_in = channels_in
        self.size_x = size_x
        self.size_y = size_y
        self.steps = 0
        self.representation = modules['representation'](num_blocks, channels_in, latent_dim).to(device)
        self.prediction = modules['prediction'](num_blocks, latent_dim, size_x, size_y, action_space_size, latent_dim).to(device)
        self.dynamics = modules['dynamics'](num_blocks, size_x, size_y,
                                 state_channels_in=latent_dim,
                                 action_channels_in=1,
                                 latent_dim=latent_dim).to(device)

        self.device = device
        self.config = config

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        image = torch.Tensor(image).to(self.device).reshape(1, self.channels_in, self.size_y, self.size_x)
        initial_state = self.representation(image)
        policy_logits, value = self.prediction(initial_state)
        return NetworkOutput(value, torch.Tensor([0]).to(self.device), policy_logits, initial_state)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # action_arr = np.zeros(self.config.action_space_size)
        # action_arr[action] = 1
        # action_arr = action_arr.reshape(1, 1, 3, 3)

        action_arr = np.ones((1, 1, self.size_y, self.size_x)) * action / self.config.action_space_size


        # dynamics + prediction function
        # hidden_state = torch.Tensor(hidden_state).to(device) <- already a tensor!
        action_tensor = torch.Tensor(action_arr).to(self.device)

        next_state, reward = self.dynamics((hidden_state, action_tensor))
        policy_logits, value = self.prediction(next_state)

        return NetworkOutput(value, reward, policy_logits, next_state)

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        # Returns the weights of this network.
        # return {k: v.cpu() for k, v in self.state_dict().items()}
        return self.state_dict()

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.steps


class SharedStorage(object):

    def __init__(self, device, config, network_factory):
        self._networks = {}
        self.device = device
        self.config = config
        self.network_factory = network_factory

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return self.network_factory(self.device, self.config)

    def save_network(self, step: int, network: Network):
        self._networks[step] = network



