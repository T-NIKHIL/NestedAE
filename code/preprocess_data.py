""" Script that calls create_preprocessed_datasets() to preprocess the data."""

import sys
import json

import click
import logging, os
logging.disable(logging.WARNING)

from inputs.dataset_inputs import list_of_nn_datasets_dict
from inputs.nn_inputs import list_of_nn_params_dict
from inputs.train_inputs import list_of_nn_train_params_dict
from utils.custom_utils import save_to_pickle, set_global_random_seed
from utils.dataset_utils import create_preprocessed_datasets

@click.command()
@click.option('--run_dir', prompt='run_dir', 
              help='The run directory contains all the NestedAE models trained on the multiscale dataset.')
@click.option('--nn_save_dir', prompt='nn_save_dir', 
              help='Specify name of the directory to store the model.')
@click.option('--nn', prompt='nn',  
              help='Specify neural network number used for making the prediction.')
@click.option('--mode', prompt='mode', 
              help='Specify whether to preprocess train or predict data.')
@click.option('--kfolds', prompt='kfolds', 
              help='Specify the number of folds for cross validation.')

def preprocess_data(run_dir, nn_save_dir, nn, mode, kfolds):

    # Make the run dir
    if not os.path.exists('../runs'):
        os.mkdir('../runs')
        print(' --> Run directory created.')
    else:
        print(' --> Found Run directory already exists.')

    nn_save_dir_path = f'../runs/{run_dir}/{nn_save_dir}'
    if os.path.exists(nn_save_dir_path) is False:
        os.mkdir(nn_save_dir_path)
        print(' --> nn directory created.')
    else:
        print(' --> nn directory already exists.')

    nn_idx = int(nn) - 1

    nn_params_dict = list_of_nn_params_dict[nn_idx]
    nn_train_params_dict = list_of_nn_train_params_dict[nn_idx]
    nn_datasets_dict = list_of_nn_datasets_dict[nn_idx]

    if len(nn_params_dict.keys()) == 0:
        raise ValueError(' --> Provided empty dictionary nn params dictionary !')

    if len(nn_train_params_dict.keys()) == 0:
        raise ValueError(' --> Provided empty dictionary nn train params dictionary !')

    if len(nn_datasets_dict.keys()) == 0:
        raise ValueError(' --> Provided empty dictionary nn datasets dictionary !')

    # Send all print statements to file for debugging
    print_file_path = nn_save_dir_path + '/' + 'preprocess_data_out.txt'
    sys.stdout = open(print_file_path, "w", encoding='utf-8')

    print(f' --> User provided command line run_dir argument : {run_dir}')
    print(f' --> User provided command line nn argument : {nn_idx}')
    print(f' --> User provided command line mode argument : {mode}')

    ################################################################################################
    # Preprocess user provided nn dictionaries
    ################################################################################################

    # Check if required keys are present in nn_train_params_dict
    _required_keys = ['global_seed',
                      'epochs', 
                      'batch_size',
                      'shuffle_data_between_epochs',
                      'optimizer',
                      'test_split']

    _required_keys_dtypes = [int, int, int, bool, dict, float]

    _provided_keys = set(list(nn_train_params_dict.keys()))

    # Check if required keys are preset in the nn dictionary
    if _provided_keys.issuperset(set(_required_keys)) is False:
        missing_keys = set(_required_keys).difference(_provided_keys)
        raise KeyError(f' --> Missing {missing_keys} in nn train params dict.')

    # Typecast entry to required collection or data type
    for i, _required_key in enumerate(_required_keys):
        if isinstance(nn_train_params_dict[_required_key], _required_keys_dtypes[i]) is False:
            raise TypeError(f' --> Value for {_required_key} key in nn_train_params_dictionary should be of type {_required_keys_dtypes[i]}.')
        
    # Perform same check for nn_params_dict
    _required_submodule_keys = ['connect_to',
                                'num_nodes_per_layer',
                                'layer_type',
                                'layer_activation',
                                'layer_kernel_init',
                                'layer_bias_init']

    _required_submodule_keys_dtypes = [list, list, list, list, list, list]

    _required_loss_keys = ['type',
                           'wt',
                           'target']

    for submodule_name, submodule_dict in \
        zip(nn_params_dict['submodules'].keys(), nn_params_dict['submodules'].values()):

        # Make sure all the required keys are there in module dictionary
        submodule_keys = set(list(submodule_dict.keys()))

        if submodule_name != 'z':

            if 'load_submodule' in submodule_keys:
                path = submodule_dict['load_submodule']
                if os.path.exists(path) is False:
                    raise FileNotFoundError(f' --> Unable to find {path} to read submodule from.')
                else:
                    continue

            if submodule_keys.issuperset(set(_required_submodule_keys)) is False:
                missing_keys = set(_required_submodule_keys).difference(submodule_keys)
                raise KeyError(f' --> Missing {missing_keys} in submodule \
                                {submodule_name} dictionary.')

            if 'loss' in submodule_keys:

                if isinstance(submodule_dict['loss'], dict) is False:
                    raise TypeError(' --> Value for "loss" key should be a dictionary.')

                loss_keys = set(list(submodule_dict['loss'].keys()))

                if loss_keys.issuperset(set(_required_loss_keys)) is False:
                    missing_keys = set(_required_loss_keys).difference(loss_keys)
                    raise KeyError(f' --> Missing {missing_keys} key in submodule \
                                    {submodule_name} loss dictionary.')

            for i, _required_key in enumerate(_required_submodule_keys):
                if isinstance(submodule_dict[_required_key], list) is False and isinstance(submodule_dict[_required_key], dict) is False:
                    raise TypeError(f' --> Value for {_required_key} key in submodule dictionary {submodule_name} should be of type list.')
                

    # Check if paths to all directories provided in nn datasets dictionary exist
    datasets = nn_datasets_dict[mode]
    for dataset_dict in list(datasets.values()):
        path = dataset_dict['path']
        if os.path.exists(path) is False:
            raise FileNotFoundError(f' --> Unable to find {path} to read dataset from.')
        
    global_seed = nn_train_params_dict['global_seed']
    set_global_random_seed(global_seed)
    print(f' --> Set global random seed {global_seed}.')
        
    # Save the input dictionaries to pickle
    save_to_pickle(list_of_nn_params_dict, 'list_of_nn_params_dict.pkl', nn_save_dir_path)
    save_to_pickle(list_of_nn_train_params_dict, 'list_of_nn_train_params_dict.pkl', nn_save_dir_path)
    save_to_pickle(list_of_nn_datasets_dict, 'list_of_nn_datasets_dict.pkl', nn_save_dir_path)

    # Save the history of all different models created in the run directory.
    with open(nn_save_dir_path + '/' + 'run_summary.txt', 'a', encoding='utf-8') as file:
        file.write(f'--NN params dict (Model {nn_idx})--' + '\n')
        file.write(json.dumps(
            list_of_nn_params_dict[nn_idx], indent=4) + '\n')
        file.write('\n')

        file.write(
            f'--NN train params dict (Model {nn_idx})--' + '\n')
        file.write(json.dumps(
            list_of_nn_train_params_dict[nn_idx], indent=4) + '\n')
        file.write('\n')

        file.write(
            f'--NN dataset dict (Model {nn_idx})--' + '\n')
        file.write(json.dumps(
            list_of_nn_datasets_dict[nn_idx], indent=4) + '\n')
        file.write('\n')
    print(' --> Saved user provided dictionaries to run_summary.txt')
        
    test_split = nn_train_params_dict['test_split']
    if int(kfolds) > 0:
        create_preprocessed_datasets(nn_save_dir_path, nn_datasets_dict, global_seed,
                                    test_split=test_split, mode=mode, kfolds=int(kfolds))
    else:
        create_preprocessed_datasets(nn_save_dir_path, nn_datasets_dict, global_seed,
                                    test_split=test_split, mode=mode)

    print(' --> Preprocessed dataset.')
    print(' --> PROGRAM EXIT.')

if __name__ == '__main__':
    preprocess_data()