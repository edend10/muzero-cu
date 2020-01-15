from comet_ml import Experiment
import json
import argparse
import io_util
import torch
import threading
import time
from muzero.util import SharedStorage, ReplayBuffer, MuZeroConfig, run_selfplay, train_network

parser = argparse.ArgumentParser()
parser.add_argument('--log_comet', type=io_util.str2bool, nargs='?', const=True, default=False, help='output directory')
parser.add_argument('--force_cpu', type=io_util.str2bool, nargs='?', const=True, default=False, help='force cpu')
parser.add_argument('--game', type=str, default='tictactoe', help='game to play')
args = parser.parse_args()

log_comet = args.log_comet
force_cpu = args.force_cpu
game_to_play = args.game

if game_to_play == 'tictactoe':
    import games.tic_tac_toe.util as game_utils
elif game_to_play == 'atari':
    import games.atari.util as game_utils
else:
    print('ERROR - Unrecognized game: "{}". Exiting...'.format(game_to_play))
    exit(1)

device_type = 'cpu' if force_cpu or not torch.cuda.is_available() else 'cuda'
print('Running with device: "{}"'.format(device_type))
device = torch.device(device_type)


experiment = None
if log_comet:
    with open('comet_config.json', 'r') as f:
        comet_config = json.load(f)

    experiment = Experiment(api_key=comet_config['api_key'],
                            project_name=comet_config['project_name'], workspace=comet_config['workspace'])

    experiment.add_tags(comet_config['bootstrap_tags'])
    experiment.log_parameter('device_type', device_type)


# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig, network_factory, test_network, test_network_params, device, experiment=None):
    storage = SharedStorage(device, config, network_factory)
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
    train_network(config, storage, replay_buffer, network_factory, test_network, test_network_params, device, experiment)

    return storage.latest_network()


def launch_job(f, *args):
    f(*args)


config = game_utils.make_config()
if log_comet:
    experiment.log_parameters(config.__dict__)


test_network_params = None
if game_to_play == 'tictactoe':
    test_network_params = dict()
    test_network_params['random'] = game_utils.random_vs_random(config, n=10000, scale_to=100)
    test_network_params['optimal'] = dict()
    test_network_params['optimal']['first'] = game_utils.optimal_vs_random(config, play_as='O', n=1000, scale_to=100)
    test_network_params['optimal']['second'] = game_utils.optimal_vs_random(config, play_as='X', n=1000, scale_to=100)
    print('random_vs_random = O: {} | X: {} | Draw: {}'.format(test_network_params['random']['O'],
                                                               test_network_params['random']['X'],
                                                               test_network_params['random']['Draw']))
    print('optimal_vs_random = O: {} | X: {} | Draw: {}'.format(test_network_params['optimal']['first']['O'],
                                                                test_network_params['optimal']['first']['X'],
                                                                test_network_params['optimal']['first']['Draw']))
    print('random_vs_optimal = O: {} | X: {} | Draw: {}'.format(test_network_params['optimal']['second']['O'],
                                                                test_network_params['optimal']['second']['X'],
                                                                test_network_params['optimal']['second']['Draw']))

network = muzero(config, game_utils.make_uniform_network,
                 game_utils.test_network, test_network_params,
                 device, experiment if log_comet else None)
