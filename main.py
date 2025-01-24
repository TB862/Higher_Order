import os
import sys
import argparse
import json
import torch
import numpy as np
import scipy.sparse as sp
from config.all_config import *
from utils import *
from train import *

# Add paths to the system
sys.path.append(os.path.abspath(os.path.join('spmpnn', 'src', 'utils')))
sys.path.append(os.path.abspath(os.path.join('spmpnn', 'src', 'models')))
sys.path.append(os.path.abspath(os.path.join('spmpnn', 'src', 'models', 'layers')))
sys.path.append(os.path.abspath(os.path.join('PathNNs_expressive', 'benchmarks')))
sys.path.append(os.path.abspath(os.path.join('grand_src')))

def list_format(string):
    try:
        return json.loads(string)  # Attempt to parse the string as JSON
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError('Input must be a JSON list')  # Raise error if input is not valid JSON

def run_experiment(pargs):
    # Unpack arguments
    config_name = pargs.config
    ds_name = pargs.dataset
    do_train = pargs.train
    do_test = pargs.test
    load = pargs.load
    model = pargs.model
    gpu = pargs.gpu
    khop = pargs.khop 
    beta_mul = pargs.beta_mul 
    nrepeat = pargs.nrepeat 

    # For sampling method ablation
    method = pargs.method

    # Collect configs, dataset
    if config_name == 'graph':  # For now, just keep it simple
        config = graph_config()
    elif isinstance(config_name, ConfigDict):
        config = config_name
    else:
        raise ValueError('Invalid configuration name')
    
    if nrepeat is not None:
        config.experiments.num_repeats = nrepeat 

    model_names = config.baselines.names

    config.baselines.MultiHop_GAT.beta_mul = beta_mul 

    ds_config = get_ds_config(config, ds_name)
    add_args = {}
    has_path_model = np.any([mn.startswith('PathNN') for mn in model_names])  # Check if PathNN in the baselines
    if has_path_model:
        add_args['path_type'] = [mn.split('_')[-1] for mn in model_names]

    # Override config file
    if method is not None:
        print('Starting run with sampling method')
        config.baselines.names = ['MultiHop_GAT']
        config.baselines.MultiHop_GAT.load_samples = True
        config.baselines.MultiHop_GAT.select_method = method
        config.baselines.MultiHop_GAT.K_hops = 3
        config.baselines.MultiHop_GAT.num_heads = [8, 1]
    if khop is not None:
        config.baselines.MultiHop_GAT.K_hops = khop 
        config.baselines.MultiHop_GAT.load_samples = True
    
    if model is not None:
        config.baselines.names = [model]
    
    if gpu is not None:
        if gpu in [0, 1, 4]:
            config.device = torch.device(f'cuda:{gpu}')
        elif gpu == -1:
            config.device = torch.device('cpu')

    data_dict = fetch_dataset(config, ds_name, False, add_args)

    # Training
    if ds_config.training.loss == 'cross entropy':
        loss = nn.CrossEntropyLoss()  # Default loss function
    else:
        raise ValueError(f'Invalid loss type {ds_config.loss} in configuration file')

    # Run training and save state
    if do_train:
        models = create_models(config, ds_config, data_dict)
        all_metrics = meta_train(config, ds_config, models, data_dict, loss)
    elif do_test:
        nrepeats = config.experiments.num_repeats
        all_metrics = load_from_checkpoint(config, ds_name, nrepeats)

    # get edge weights of GAT 
    data = data_dict['dataset']
    x = data.x
    edge_index = data.edge_index 

    # convert to coo matrix 
    """out, (edge_index, alpha) = models['GAT'](x, edge_index, return_attention_weights=True)
    print(alpha.shape)
    num_nodes = torch.max(edge_index) + 1 
    alpha = to_numpy(alpha).squeeze()
    edge_index = to_numpy(edge_index) 
    row, col = edge_index[0], edge_index[1]

    coo_adj_matrix = sp.coo_matrix((alpha, (row, col)), shape=(num_nodes, num_nodes))
    sp.save_npz(f'{ds_name}_adj.npz', coo_adj_matrix)
    
    exit()"""

    print(all_metrics)
    for model, metric_dict in all_metrics.items():
        acc = metric_dict['Accuracy']
        print(f'{ds_name} result with method {method}')
        print(model, np.mean(acc), np.std(acc))

    # Optional significance testing
    if do_train and do_test:
        pass

    return all_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='The name of the dataset to run')
    parser.add_argument('--config', type=str, help='The type of configuration file', default='graph')
    parser.add_argument('--gpu', type=int, default=None, help='Override model GPU in config file')
    parser.add_argument('--khop', type=int, default=None, help='Override model GPU in config file')
    parser.add_argument('--beta_mul', type=float, default=1.0, help='Override model GPU in config file')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint directory for loading models', default='checkpoints')
    parser.add_argument('--train', action='store_true', help='Option to train the model')
    parser.add_argument('--test', action='store_true', help='Option to test the model')
    parser.add_argument('--load', type=list_format, nargs='+', default=None, help='Load models from list')
    parser.add_argument('--method', type=str, default=None, help='Sampling method ablation')
    parser.add_argument('--model', type=str, default=None, help='Override model in config file')
    parser.add_argument('--nrepeat', type=int, default=None, help='Override model GPU in config file')

    pargs = parser.parse_args()
    run_experiment(pargs)