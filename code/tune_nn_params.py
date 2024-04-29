""" NN parameter tuning using wandb"""
import time
import os
import sys
import copy
import json
import warnings
from joblib import Parallel, delayed
from torch import load, get_num_threads, get_num_interop_threads
from torch.utils.data import DataLoader
import click
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

# User defined libraries
from utils.dataset_utils import create_preprocessed_datasets
from utils.custom_utils import save_to_pickle, set_global_random_seed
from utils.nn_utils import create_callback_object
from inputs.train_inputs import list_of_nn_train_params_dict
from inputs.dataset_inputs import list_of_nn_datasets_dict
from inputs.nn_inputs import list_of_nn_params_dict
from nn.vanilla_ae import VanillaAE
from wandb_api_key import api_key
import os

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

def run_wandb_agent(fold_num, train_dataloader, val_dataloader, accelerator):
    """
    Runs the wandb agent for tuning neural network parameters.

    Args:
        nn_params_dict (dict): Dictionary containing the neural network parameters.
        nn_train_params_dict (dict): Dictionary containing the training parameters for the neural network.
        nn_datasets_dict (dict): Dictionary containing the datasets for training and validation.
        nn_save_dir (str): Directory to save the neural network model.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        accelerator (str): Accelerator to use for training.

    Returns:
        None
    """
    # To start a wandb run have to call wandb.init(). At any time there can only be one active run. To finish have to call run.finish()
    run = wandb.init(job_type='training', resume=False, reinit=False, dir=nn_save_dir+'/tune_nn_params')
    new_nn_params_dict = copy.deepcopy(nn_params_dict)
    for submodule in new_nn_params_dict['submodules'].keys():
        new_nn_params_dict['submodules'][submodule]['num_nodes_per_layer'] = [
            wandb.config[f'{submodule}-hidden_dim']]*wandb.config[f'{submodule}-hidden_layers'] + [wandb.config[f'{submodule}-output_dim']]
        new_nn_params_dict['submodules'][submodule]['layer_type'] = [
            wandb.config[f'{submodule}-layer_type']]*(wandb.config[f'{submodule}-hidden_layers'] + 1)
        new_nn_params_dict['submodules'][submodule]['layer_activation'] = [
            wandb.config[f'{submodule}-layer_activation']]*(wandb.config[f'{submodule}-hidden_layers'] + 1)
        new_nn_params_dict['submodules'][submodule]['layer_kernel_init'] = [
            wandb.config[f'{submodule}-layer_kernel_init']]*(wandb.config[f'{submodule}-hidden_layers'] + 1)
        new_nn_params_dict['submodules'][submodule]['layer_kernel_init_gain'] = [
            wandb.config[f'{submodule}-layer_kernel_init_gain']]*(wandb.config[f'{submodule}-hidden_layers'] + 1)
        new_nn_params_dict['submodules'][submodule]['layer_bias_init'] = [
            wandb.config[f'{submodule}-layer_bias_init']]*(wandb.config[f'{submodule}-hidden_layers'] + 1)
        new_nn_params_dict['submodules'][submodule][
            'layer_weight_reg_l1'] = wandb.config[f'{submodule}-layer_weight_reg_l1']
        new_nn_params_dict['submodules'][submodule][
            'layer_weight_reg_l2'] = wandb.config[f'{submodule}-layer_weight_reg_l2']
        new_nn_params_dict['submodules'][submodule]['save_params'] = False
        new_nn_params_dict['submodules'][submodule]['save_output_on_fit_end'] = False
        new_nn_params_dict['submodules'][submodule]['save_output_on_epoch_end'] = False
        # TODO : Can also include dropout here

    # Remove the keys hidden_dim, hidden_layers, output_dim from new_nn_params_dict
    for submodule in new_nn_params_dict['submodules'].keys():
        new_nn_params_dict['submodules'][submodule].pop('hidden_dim', None)
        new_nn_params_dict['submodules'][submodule].pop('hidden_layers', None)
        new_nn_params_dict['submodules'][submodule].pop('output_dim', None)

    # Get the fold number from the sweep config dictionary

    model_sweep_dir = nn_save_dir + '/tune_nn_params' + '/' + run.name + '_fold_' + str(fold_num)

    # Create a run directory under tune_nn_params
    if not os.path.exists(model_sweep_dir):
        os.makedirs(model_sweep_dir)

    # Send all print statements to file for debugging
    print_file_path = model_sweep_dir + '/' + 'train_out.txt'
    sys.stdout = open(print_file_path, "w", encoding='utf-8')

    # Build the nn model
    ae = VanillaAE(nn_save_dir, new_nn_params_dict, nn_train_params_dict, nn_datasets_dict)
    ae.compile()
    print(' --> Model Compilation step complete.')

    callbacks = create_callback_object(nn_train_params_dict, model_sweep_dir)

    trainer = Trainer(max_epochs=nn_train_params_dict['epochs'],
                      accelerator=accelerator,
                      deterministic=True,
                      logger=WandbLogger(),
                      callbacks=callbacks,
                      enable_model_summary=False,
                      enable_progress_bar=False)
    
    trainer.fit(model=ae, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=None)
    run.finish()
    time.sleep(5)
    # Move the files from wandb directory to model sweep directory
    os.system(f'cp -r {nn_save_dir}/tune_nn_params/wandb/*-{run.id}/* {model_sweep_dir}')
    # Delete the run directory from wandb directory
    os.system(f'rm -r {nn_save_dir}/tune_nn_params/wandb/*-{run.id}')
    print(' --> EXIT.')

def run_parallel_kfold(fold_num, sweep_config, trials_in_sweep, accelerator):
    # Create a copy of sweep_config
    sweep_config_for_fold = copy.deepcopy(sweep_config)
    # Check if fold is present in the sweep_config name
    if 'fold' in sweep_config_for_fold['name']:
        sweep_config_for_fold['name'] = sweep_config_for_fold['name'].split('_fold')[0]
    sweep_config_for_fold['name'] = sweep_config_for_fold['name'] + '_fold_' + str(fold_num)
    sweep_id = wandb.sweep(sweep_config_for_fold)
    if int(trials_in_sweep) == -1:
        count = None
    else:
        count = int(trials_in_sweep)
    # Load the train and validation datsets for the fold
    train_dataset = load(
        nn_save_dir + '/datasets' + f'/train_dataset_fold_{fold_num}.pt')
    train_dataloader = DataLoader(train_dataset,
                                    batch_size=nn_train_params_dict['batch_size'],
                                    shuffle=nn_train_params_dict['shuffle_data_between_epochs'],
                                    num_workers=0)
    val_dataset = load(
        nn_save_dir + '/datasets' + f'/val_dataset_fold_{fold_num}.pt')
    val_dataloader = DataLoader(val_dataset,
                                batch_size=nn_train_params_dict['batch_size'],
                                shuffle=False,
                                num_workers=0)
    wandb.agent(sweep_id, lambda: run_wandb_agent(fold_num, train_dataloader, val_dataloader, accelerator), count=count)
    

@click.command()
@click.option('--run_dir', prompt='run_dir',
            help='Specify the run directory.')
@click.option('--nn', prompt='nn',
            help='Specify the neural network to tune.')
@click.option('--kfolds', prompt='kfolds',
            help='Specify the number of folds for kfold cross validation.')
@click.option('--user_name', prompt='user_name',
            help='Specify the wandb username.')
@click.option('--project_name', prompt='project_name',
            help='Specify the wandb project name.')
@click.option('--sweep_type', prompt='sweep_type',
            help='Specify the sweep type.')
@click.option('--metric', prompt='metric',
            help='Specify the metric to optimize.')
@click.option('--goal', prompt='goal',
            help='Specify the goal to optimize.')
@click.option('--trials_in_sweep', prompt='trials_in_sweep',
            help='Specify the number of trials in the sweep.')
@click.option('--accelerator', prompt='accelerator',
            help='Specify the accelerator to use.')
def run_tune_nn_params(run_dir, nn, kfolds, user_name, project_name, sweep_type, metric, goal, trials_in_sweep, accelerator):
    """
    Runs the hyperparameter tuning process for a neural network model.

    Args:
        run_dir (str): The directory to store the hyperparameter runs.
        nn (str): The neural network model to tune.
        kfolds (str): The number of folds for k-fold cross validation.
        user_name (str): The username for the project.
        project_name (str): The name of the project.
        sweep_type (str): The type of sweep configuration.
        metric (str): The metric to optimize during tuning.
        goal (str): The goal of the metric (e.g., 'minimize' or 'maximize').
        trials_in_sweep (str): The number of trials to run in each sweep.
        accelerator (str): The accelerator to use for training.

    Returns:
        None
    """
    os.environ["WANDB_API_KEY"] = api_key
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_ENTITY"] = user_name

    global nn_params_dict
    global nn_train_params_dict
    global nn_datasets_dict
    global nn_save_dir
    global tune_nn_params_dir

    nn_idx = int(nn) - 1
    # Create a directory to store the hyperparameter runs
    run_dir = '../runs'+'/'+run_dir
    nn_save_dir = run_dir+'/'+list_of_nn_params_dict[nn_idx]['model_type']
    tune_nn_params_dir = nn_save_dir + '/tune_nn_params'
    if not os.path.exists(tune_nn_params_dir):
        os.makedirs(tune_nn_params_dir)

    nn_params_dict = list_of_nn_params_dict[nn_idx]
    nn_train_params_dict = list_of_nn_train_params_dict[nn_idx]
    nn_datasets_dict = list_of_nn_datasets_dict[nn_idx]

    # Set the  global random seed
    set_global_random_seed(nn_train_params_dict['global_seed'])

    # Send all print statements to file for debugging
    print_file_path = nn_save_dir + '/' + 'preprocess_data_out.txt'
    sys.stdout = open(print_file_path, "w", encoding='utf-8')

    if int(kfolds) > 0:
        create_preprocessed_datasets(nn_save_dir, nn_datasets_dict, nn_train_params_dict['global_seed'],
                                     test_split=nn_train_params_dict['test_split'], mode='train', kfolds=int(kfolds))
    else:
        create_preprocessed_datasets(nn_save_dir, nn_datasets_dict, nn_train_params_dict['global_seed'],
                                     test_split=nn_train_params_dict['test_split'], mode='train')

    params_to_tune = {}
    # Check for each submodule in nn_params_dict if a dictionary exists for any of the keys and add to sweep config.
    for submodule in nn_params_dict['submodules'].keys():
        submodule_params = copy.deepcopy(list(nn_params_dict['submodules'][submodule].keys()))
        submodule_params.remove('connect_to')
        if 'loss' in submodule_params:
            submodule_params.remove('loss')
        for submodule_param in submodule_params:
            params_to_tune[f'{submodule}-{submodule_param}'] = nn_params_dict['submodules'][submodule][submodule_param]

    # Define the sweep configuration
    sweep_config = {
        'name': list_of_nn_params_dict[nn_idx]['model_type'],
        'method': sweep_type,
        'metric': {
            'name': metric,
            'goal': goal
        },
        'parameters': params_to_tune
    }

    # The for loop is embarassingly parallel and can be run on multiple cores via multiprocessing.
    Parallel(n_jobs=int(kfolds), backend='loky')(delayed(run_parallel_kfold)(fold_num, sweep_config, trials_in_sweep, accelerator) for fold_num in range(int(kfolds)))

if __name__ == '__main__':
    nn_save_dir = None
    tune_nn_params_dir = None
    nn_params_dict = None
    nn_train_params_dict = None
    nn_datasets_dict = None
    run_tune_nn_params()

# Models for film scale perovskite dataset v2 trained using 90:10 split and batch size 10

# paths_seed10 = ['nthota2/perovskite_dataset_v2/i0siejd3',
#                 'nthota2/perovskite_dataset_v2/41lyo4zx',
#                 'nthota2/perovskite_dataset_v2/090ev6lt',
#                 'nthota2/perovskite_dataset_v2/4hx3g1ze',
#                 'nthota2/perovskite_dataset_v2/5lxjh9yz',
#                 'nthota2/perovskite_dataset_v2/plfhcevp',
#                 'nthota2/perovskite_dataset_v2/s2blmq1p',
#                 'nthota2/perovskite_dataset_v2/33kw47zq',
#                 'nthota2/perovskite_dataset_v2/51g6co9z',
#                 'nthota2/perovskite_dataset_v2/ea6e0vvg',
#                 'nthota2/perovskite_dataset_v2/s1xzk0v8']

# paths_seed1 = ['nthota2/perovskite_dataset_v2/661g95rw',
#                'nthota2/perovskite_dataset_v2/0g449pn8',
#                'nthota2/perovskite_dataset_v2/lqh9dg48',
#                'nthota2/perovskite_dataset_v2/e7dgdq6i',
#                'nthota2/perovskite_dataset_v2/26a8ng2u',
#                'nthota2/perovskite_dataset_v2/ks0zvbhp',
#                'nthota2/perovskite_dataset_v2/adx91d6o',
#                'nthota2/perovskite_dataset_v2/901war6q',
#                'nthota2/perovskite_dataset_v2/qwiryh15',
#                'nthota2/perovskite_dataset_v2/krs81m9u',
#                'nthota2/perovskite_dataset_v2/319183bd']

# paths_seed2 = ['nthota2/perovskite_dataset_v2/mdshovqj',
#                'nthota2/perovskite_dataset_v2/y3v0p2x1',
#                'nthota2/perovskite_dataset_v2/st18wn73',
#                'nthota2/perovskite_dataset_v2/t1ftdjr5',
#                'nthota2/perovskite_dataset_v2/041zbojc',
#                'nthota2/perovskite_dataset_v2/1v79q4r9',
#                'nthota2/perovskite_dataset_v2/00tu85tg',
#                'nthota2/perovskite_dataset_v2/4wx9bbt3',
#                'nthota2/perovskite_dataset_v2/2v4pf9yo',
#                'nthota2/perovskite_dataset_v2/jirnsqq2',
#                'nthota2/perovskite_dataset_v2/8bgcj3id']

# paths_seed3 = ['nthota2/perovskite_dataset_v2/9c91pcxl',
#                'nthota2/perovskite_dataset_v2/iisquapr',
#                'nthota2/perovskite_dataset_v2/f33cluje',
#                'nthota2/perovskite_dataset_v2/kmx635x0',
#                'nthota2/perovskite_dataset_v2/zcohntic',
#                'nthota2/perovskite_dataset_v2/nx650722',
#                'nthota2/perovskite_dataset_v2/yk3x5pn5',
#                'nthota2/perovskite_dataset_v2/lt3yut3v',
#                'nthota2/perovskite_dataset_v2/uwni8be0',
#                'nthota2/perovskite_dataset_v2/9cpliop8',
#                'nthota2/perovskite_dataset_v2/ptophhdq']

# paths_seed4 = ['nthota2/perovskite_dataset_v2/g5som797',
#                'nthota2/perovskite_dataset_v2/fgz3igxx',
#                'nthota2/perovskite_dataset_v2/l3q1w43r',
#                'nthota2/perovskite_dataset_v2/3shs46xm',
#                'nthota2/perovskite_dataset_v2/iscztzx0',
#                'nthota2/perovskite_dataset_v2/aeqbbo34',
#                'nthota2/perovskite_dataset_v2/hful22lx',
#                'nthota2/perovskite_dataset_v2/31bx1k6h',
#                'nthota2/perovskite_dataset_v2/b7lr6u9d',
#                'nthota2/perovskite_dataset_v2/9tv3phh1',
#                'nthota2/perovskite_dataset_v2/oioxfv5e']