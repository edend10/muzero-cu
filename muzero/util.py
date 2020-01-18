import math
from typing import List
import numpy as np
import torch
from muzero.classes import MinMaxStats, MuZeroConfig, Action, Player, Node,\
    ActionHistory, Game, ReplayBuffer, NetworkOutput, Network, SharedStorage, KnownBounds


def make_board_game_config(action_space_size: int, max_moves: int,
                           dirichlet_alpha: float,
                           lr_init: float,
                           init_env,
                           get_env_legal_actions,
                           get_env_obs,
                           get_to_play,
                           turn_based,
                           make_uniform_network,
                           optimize_reward=True) -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if num_moves >= 0:
            return 1.0
        else:
            return 0.0  # Play according to the max.

    return MuZeroConfig(
        action_space_size=action_space_size,
        max_moves=max_moves,
        discount=1.0,
        dirichlet_alpha=dirichlet_alpha,
        num_simulations=10,
        batch_size=32,
        td_steps=max_moves,  # Always use Monte Carlo return.
        num_actors=1,
        lr_init=lr_init,
        lr_decay_steps=400e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=KnownBounds(-1, 1),
        init_env=init_env,
        get_env_legal_actions=get_env_legal_actions,
        get_env_obs=get_env_obs,
        get_to_play=get_to_play,
        turn_based=turn_based,
        make_uniform_network=make_uniform_network,
        optimize_reward=optimize_reward
    )


def make_go_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=362, max_moves=722, dirichlet_alpha=0.03, lr_init=0.01)


def make_chess_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=4672, max_moves=512, dirichlet_alpha=0.3, lr_init=0.1)


def make_shogi_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=11259, max_moves=512, dirichlet_alpha=0.15, lr_init=0.1)


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: MuZeroConfig,
                 storage: SharedStorage,
                 replay_buffer: ReplayBuffer,
                 device):
    game_counter = 0
    with torch.no_grad():
        while True:
            network = config.make_uniform_network(device, config)
            network.set_weights(storage.latest_network().get_weights())
            network.eval()
            game = play_game(config, network)
            replay_buffer.save_game(game)

            game_counter += 1
            if game_counter % 1000 == 0 or game_counter == 1:
                print('Worker so far played {} games...'.format(game_counter))


def mcts_action(config, network, game):
    root = Node(0)
    current_observation = game.make_image()
    expand_node(root, game.to_play(), game.legal_actions(),
                network.initial_inference(current_observation))

    add_exploration_noise(config, root)

    # We then run a Monte Carlo Tree Search using only action sequences and the
    # model learned by the network.
    run_mcts(config, root, game.action_history(), network)
    return root, select_action(config, len(game.history), root, network)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network) -> Game:
    game = config.new_game(play_as='O')

    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.

        root, action = mcts_action(config, network, game)

        game.apply(action)
        game.store_search_statistics(root)
        game.switch_player()

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
            action_index, node = select_child(config, node, min_max_stats)
            history.add_action(Action(action_index))
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state,
                                                     history.last_action().index)
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(search_path, network_output.value.item(), history.to_play(),
                      config.discount, min_max_stats)


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
        node.value_sum += value # if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MuZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def select_action(config: MuZeroConfig, num_moves: int, node: Node,
                  network: Network):
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps())
    action = softmax_sample(visit_counts, t)
    return Action(action)


def update_weights(optimizer: torch.optim.Optimizer, network: Network, batch, step, config, device, experiment=None):
    network.train()

    mse_criterion = torch.nn.MSELoss()
    # cross_entropy_criterion = torch.nn.CrossEntropyLoss()

    # reward_loss = 0
    value_loss, reward_loss, policy_loss = 0, 0, 0
    for image, actions, targets in batch:
        # Initial step, from the real observation.
        value, reward, policy_logits, hidden_state = network.initial_inference(image)
        predictions = [(1.0, value, reward, policy_logits)]

        # Recurrent steps, from action and previous hidden state.
        for action in actions:
            value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action.index)
            predictions.append((1.0 / len(actions), value, reward, policy_logits))

            # TODO: no need to scale gradients in half?

        for prediction, target in zip(predictions, targets):
            gradient_scale, value, reward, policy_logits = prediction
            target_value, target_reward, target_policy = target

            if len(target_policy) > 0:
                # reward_loss += mse_criterion(torch.Tensor([target_reward]).to(device), reward)
                value_loss += gradient_scale * mse_criterion(torch.Tensor([target_value]).to(device), value)
                if config.optimize_reward:
                    reward_loss += gradient_scale * mse_criterion(torch.Tensor([target_reward]).to(device), reward)
                # policy_loss += gradient_scale * cross_entropy_criterion(policy_logits.reshape(1, -1),
                #                                        torch.LongTensor([np.argmax(target_policy)]).to(device))
                policy_loss += gradient_scale * -(torch.log_softmax(policy_logits.reshape(1, -1), dim=1) * torch.Tensor(target_policy).to(device)).sum()

            else:
                # absorbing state passed end of game
                pass

    total_loss = value_loss + policy_loss
    if config.optimize_reward:
        total_loss += reward_loss

    optimizer.zero_grad()
    total_loss.backward()
    # torch.nn.utils.clip_grad_norm_(network.parameters(), 10)
    optimizer.step()

    if step % config.train_report_interval == 0:
        # reporting
        avg_policy_loss = policy_loss.item() / len(batch)
        # avg_reward_loss = reward_loss.item() / len(batch)
        avg_value_loss = value_loss.item() / len(batch)
        avg_total_loss = total_loss.item() / len(batch)

        print('step {} - total_loss: {} | policy loss: {} | value loss: {}'
              .format(network.steps, avg_total_loss, avg_policy_loss, avg_value_loss))

        if experiment:
            experiment.log_metric('policy_loss', avg_policy_loss, step=network.steps)
            # experiment.log_metric('reward_loss', avg_reward_loss, step=network.steps)
            experiment.log_metric('value_loss', avg_value_loss, step=network.steps)
            experiment.log_metric('total_loss', avg_total_loss, step=network.steps)

    network.steps += 1


# Stubs to make the typechecker happy.
def softmax_sample(visit_counts, temperature: float):
    distribution = np.array([x[0] for x in visit_counts]) ** temperature
    p_sum = distribution.sum()
    sample_temp = distribution / p_sum
    # selected_action_index = np.argmax(np.random.multinomial(1, sample_temp, 1))
    selected_action_index = np.argmax(sample_temp)
    return visit_counts[selected_action_index][1]


def train_network(config: MuZeroConfig,
                  storage: SharedStorage,
                  replay_buffer: ReplayBuffer,
                  network_factory,
                  test_network,
                  test_network_params,
                  device,
                  experiment=None):
    # network = Network()
    network = network_factory(device, config)
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=config.lr_init,
                                weight_decay=config.lr_decay_rate,
                                momentum=config.momentum)
    # optimizer = torch.optim.Adam(network.parameters(), lr=config.lr_init, weight_decay=config.lr_decay_rate)

    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        if test_network and i % config.test_interval == 0:
            test_network(config, network, test_network_params, experiment)

        # for debugging
        if i == 1 or (i+1) % 100 == 0:
            params = list(network.prediction.parameters())[0]

            avg_params = torch.mean(params).item()
            avg_params_grad = torch.mean(params.grad).item()

            if experiment:
                experiment.log_metric('avg_network_weights', avg_params)
                experiment.log_metric('avg_network_gradients', avg_params_grad)

        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        update_weights(optimizer, network, batch, i, config, device, experiment)

    storage.save_network(config.training_steps, network)



