import numpy as np
import gym
from muzero.classes import Network, MuZeroConfig
from muzero.util import mcts_action
import games.atari.models


# TODO: input to representation should be last k frames (32?)


def make_config() -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if training_steps < 500e3:
            return 1.0
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25

    return MuZeroConfig(
        action_space_size=7,
        max_moves=10, #27000,  # Half an hour at action repeat 4.
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=10,
        batch_size=32,
        td_steps=10,
        num_actors=1,
        lr_init=0.05,
        lr_decay_steps=350e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        init_env=lambda: gym.make('Assault-v0'),
        get_env_legal_actions=lambda env: list(range(env.action_space.n)),
        get_env_obs=lambda obs: np.rollaxis(obs, 2, 0) / 256,
        get_to_play=lambda env: 'O',
        turn_based=False,
        optimize_reward=True
    )


def network_vs_atari(config, network, n=100):
    results = []
    for i in range(n):
        game = config.new_game()

        while not game.terminal():
            _, action = mcts_action(config, network, game)
            game.apply(action)

        results.append(sum(game.rewards))

    return sum(results) / len(results)


def test_network(config, network, _, experiment=None):
    network.eval()
    result = network_vs_atari(config, network)
    print('#' * 156)
    print('## {0: <150} ##'.format('Step {} test results'.format(network.steps)))
    print('## {0: <150} ##'.format(
        'avg_final_score: {:.4f} '.format(result)))
    print('#' * 156)

    if experiment:
        experiment.log_metric('avg_final_score', result, step=network.steps)


def make_uniform_network(device, config):
    modules = {
        'representation': games.atari.models.Representation,
        'prediction': games.atari.models.Prediction,
        'dynamics': games.atari.models.Dynamics
    }

    return Network(modules=modules,
                   num_blocks=1,
                   channels_in=3,
                   size_x=160,
                   size_y=250,
                   latent_dim=32,
                   action_space_size=config.action_space_size,
                   device=device,
                   config=config)


