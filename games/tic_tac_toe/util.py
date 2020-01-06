import numpy as np
from gym_tictactoe.env import TicTacToeEnv, check_game_status
import games.tic_tac_toe.models
from muzero.util import Network, make_board_game_config, mcts_action


def make_config():
    return make_board_game_config(
        action_space_size=9,
        max_moves=9,
        dirichlet_alpha=0.05,
        lr_init=0.01,
        init_env=lambda: TicTacToeEnv(),
        get_env_legal_actions=lambda env: [pos for pos in range(len(env.board)) if env.board[pos] == 0],
        get_env_obs=lambda obs: np.array(obs[0]),
        get_to_play=lambda env: env.mark,
        turn_based=True,
        optimize_reward=False
    )


def random_vs_random(config, n=10000, scale_to=None):
    # results should be biased in favor of 'O' player as it always starts
    results = {'O': 0, 'X': 0, 'Draw': 0}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        game = config.new_game()
        while not game.terminal():
            action = np.random.choice(game.legal_actions())
            game.apply(action)
            turn = not turn

        game_result = check_game_status(game.environment.board)

        if game_result == 1:
            r = 'O'
        elif game_result == 2:
            r = 'X'
        else:
            r = 'Draw'

        results[r] += 1

    if scale_to:
        for key in results.keys():
            results[key] = results[key] * (scale_to / n)

    return results


def network_vs_random(config, network, play_as='O', n=100):
    # network against random
    results = {'O': 0, 'X': 0, 'Draw': 0}
    for i in range(n):
        first_turn = play_as == 'O'
        turn = first_turn
        game = config.new_game()

        while not game.terminal():
            if turn:
                _, action = mcts_action(config, network, game)
            else:
                action = np.random.choice(game.legal_actions())
            game.apply(action)
            turn = not turn

        game_result = check_game_status(game.environment.board)

        if game_result == 1:
            r = 'O'
        elif game_result == 2:
            r = 'X'
        else:
            r = 'Draw'

        results[r] += 1
    return results


def test_network(config, network, random_baseline_results, experiment=None):
    network.eval()

    O_results = network_vs_random(config, network, play_as='O')
    X_results = network_vs_random(config, network, play_as='X')

    O_win_rate = O_results['O'] / random_baseline_results['O'] if random_baseline_results['O'] != 0 else np.inf
    X_win_rate = X_results['X'] / random_baseline_results['X'] if random_baseline_results['X'] != 0 else np.inf
    avg_win_rate = (O_win_rate + X_win_rate) / 2
    O_nonlose_rate = (O_results['O'] + O_results['Draw']) / (
            random_baseline_results['O'] + random_baseline_results['Draw']) if (random_baseline_results['O'] +
                                                                                random_baseline_results[
                                                                                    'Draw']) != 0 else np.inf
    X_nonlose_rate = (X_results['X'] + X_results['Draw']) / (
            random_baseline_results['X'] + random_baseline_results['Draw']) if (random_baseline_results['X'] +
                                                                                random_baseline_results[
                                                                                    'Draw']) != 0 else np.inf
    avg_nonlose_rate = (O_nonlose_rate + X_nonlose_rate) / 2

    print('#' * 156)
    print('## {0: <150} ##'.format('Step {} test results'.format(network.steps)))
    print('## {0: <150} ##'.format(
        'random_vs_random = X: {} | O: {} | Draw: {}'.format(random_baseline_results['X'], random_baseline_results['O'],
                                                             random_baseline_results['Draw'])))
    print('## {0: <150} ##'.format(
        'O_vs_random = X: {} | O: {} | Draw: {} '.format(O_results['X'], O_results['O'], O_results['Draw'])))
    print('## {0: <150} ##'.format(
        'X_vs_random = X: {} | O: {} | Draw: {}'.format(X_results['X'], X_results['O'], X_results['Draw'])))
    print('## {0: <150} ##'.format('-' * 150))
    print('## {0: <150} ##'.format(
        'avg win rate: {} | O win rate: {} | X win rate: {}'.format(avg_win_rate, O_win_rate, X_win_rate)))
    print('## {0: <150} ##'.format(
        'avg non-lose rate: {} | O non-lose rate: {} | X non-lose rate: {}'.format(avg_nonlose_rate, O_nonlose_rate,
                                                                                   X_nonlose_rate)))
    print('#' * 156)

    if experiment:
        experiment.log_metric('avg_win_rate', avg_win_rate, step=network.steps)
        experiment.log_metric('O_win_rate', O_win_rate, step=network.steps)
        experiment.log_metric('X_win_rate', X_win_rate, step=network.steps)
        experiment.log_metric('avg_win_rate', avg_win_rate, step=network.steps)
        experiment.log_metric('O_nonlose_rate', O_nonlose_rate, step=network.steps)
        experiment.log_metric('X_nonlose_rate', X_nonlose_rate, step=network.steps)
        experiment.log_metric('avg_nonlose_rate', avg_nonlose_rate, step=network.steps)


def make_uniform_network(device, config):
    modules = {
        'representation': games.tic_tac_toe.models.Representation,
        'prediction': games.tic_tac_toe.models.Prediction,
        'dynamics': games.tic_tac_toe.models.Dynamics
    }

    return Network(modules=modules,
                   num_blocks=1,
                   channels_in=1,
                   size_x=3,
                   size_y=3,
                   latent_dim=9,
                   action_space_size=config.action_space_size,
                   device=device,
                   config=config)


