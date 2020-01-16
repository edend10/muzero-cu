import numpy as np
import torch
from gym_tictactoe.env import TicTacToeEnv, check_game_status
import games.tic_tac_toe.models
import games.tic_tac_toe.models_gcn
from games.tic_tac_toe.optimal_player_util import get_optimal_action
from muzero.util import Network, make_board_game_config, mcts_action


def make_config():
    return make_board_game_config(
        action_space_size=9,
        max_moves=9,
        dirichlet_alpha=0.05,
        lr_init=0.001,
        init_env=lambda: TicTacToeEnv(),
        get_env_legal_actions=lambda env: [pos for pos in range(len(env.board)) if env.board[pos] == 0],
        get_env_obs=lambda obs: np.array(obs[0]),
        get_to_play=lambda env: env.mark,
        turn_based=True,
        optimize_reward=False,
        make_uniform_network=make_uniform_network
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


def network_vs_random(config, network, play_as='O', n=1000, scale_to=100):
    # network against random
    results = {'O': 0, 'X': 0, 'Draw': 0}
    for i in range(n):
        first_turn = play_as == 'O'
        turn = first_turn
        game = config.new_game()

        while not game.terminal():
            if turn:
                _, action = mcts_action(config, network, game, exploration_noise=False)
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

    if scale_to:
        for key in results.keys():
            results[key] = results[key] * (scale_to / n)

    return results


def optimal_vs_random(config, play_as='O', n=1000, scale_to=100):
    # network against random
    results = {'O': 0, 'X': 0, 'Draw': 0}
    for i in range(n):
        first_turn = play_as == 'O'
        turn = first_turn
        game = config.new_game()

        while not game.terminal():
            if turn:
                action = get_optimal_action(game.environment.board)
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

    if scale_to:
        for key in results.keys():
            results[key] = results[key] * (scale_to / n)

    return results


def lift_over_baseline(name, O_results, X_results, baseline_results_O, baseline_results_X=None, experiment=None, steps=None):
    if not baseline_results_X:
        baseline_results_X = baseline_results_O

    O_win_rate = O_results['O'] / baseline_results_O['O'] if baseline_results_O['O'] != 0 else np.inf
    X_win_rate = X_results['X'] / baseline_results_X['X'] if baseline_results_X['X'] != 0 else np.inf
    avg_win_rate = (O_win_rate + X_win_rate) / 2

    O_nonlose_rate = (O_results['O'] + O_results['Draw']) / (
            baseline_results_O['O'] + baseline_results_O['Draw']) if (baseline_results_O['O'] +
                                                                      baseline_results_O[
                                                                                    'Draw']) != 0 else np.inf
    X_nonlose_rate = (X_results['X'] + X_results['Draw']) / (
            baseline_results_X['X'] + baseline_results_X['Draw']) if (baseline_results_X['X'] +
                                                                      baseline_results_X['Draw']) != 0 else np.inf
    avg_nonlose_rate = (O_nonlose_rate + X_nonlose_rate) / 2

    print('#' * 156)
    print('## {0: <150} ##'.format('Step {} test results - {}'.format(steps, name)))
    print('## {0: <150} ##'.format(
        '{}_vs_random = O: {} | X: {} | Draw: {}'.format(name, baseline_results_O['O'],
                                                             baseline_results_O['X'],
                                                             baseline_results_O['Draw'])))

    if name is not 'random':
        print('## {0: <150} ##'.format(
            'random_vs_{}= O: {} | X: {} | Draw: {}'.format(name, baseline_results_X['O'],
                                                                  baseline_results_X['X'],
                                                                  baseline_results_X['Draw'])))

    print('## {0: <150} ##'.format(
        'O_vs_random = O: {} | X: {} | Draw: {} '.format(O_results['O'], O_results['X'], O_results['Draw'])))
    print('## {0: <150} ##'.format(
        'X_vs_random = O: {} | X: {} | Draw: {}'.format(X_results['O'], X_results['X'], X_results['Draw'])))
    print('## {0: <150} ##'.format('-' * 150))
    print('## {0: <150} ##'.format(
        'avg win rate: {} | O win rate: {} | X win rate: {}'.format(avg_win_rate, O_win_rate, X_win_rate)))
    print('## {0: <150} ##'.format(
        'avg non-lose rate: {} | O non-lose rate: {} | X non-lose rate: {}'.format(avg_nonlose_rate, O_nonlose_rate,
                                                                                   X_nonlose_rate)))
    print('#' * 156)

    if experiment:
        experiment.log_metric('{}_avg_win_rate'.format(name), avg_win_rate, step=steps)
        experiment.log_metric('{}_O_win_rate'.format(name), O_win_rate, step=steps)
        experiment.log_metric('{}_X_win_rate'.format(name), X_win_rate, step=steps)
        experiment.log_metric('{}_avg_win_rate'.format(name), avg_win_rate, step=steps)
        experiment.log_metric('{}_O_nonlose_rate'.format(name), O_nonlose_rate, step=steps)
        experiment.log_metric('{}_X_nonlose_rate'.format(name), X_nonlose_rate, step=steps)
        experiment.log_metric('{}_avg_nonlose_rate'.format(name), avg_nonlose_rate, step=steps)


def test_network(config, network, test_network_params, experiment=None):
    with torch.no_grad():
        network.eval()

        print('testing network...')

        O_results = network_vs_random(config, network, play_as='O')
        X_results = network_vs_random(config, network, play_as='X')

        lift_over_baseline(name='random',
                           O_results=O_results,
                           X_results=X_results,
                           baseline_results_O=test_network_params['random'],
                           baseline_results_X=test_network_params['random'],
                           experiment=experiment,
                           steps=network.steps)

        lift_over_baseline(name='optimal',
                           O_results=O_results,
                           X_results=X_results,
                           baseline_results_O=test_network_params['optimal']['first'],
                           baseline_results_X=test_network_params['optimal']['second'],
                           experiment=experiment,
                           steps=network.steps)


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
                   latent_dim=1,
                   action_space_size=config.action_space_size,
                   device=device,
                   config=config)

