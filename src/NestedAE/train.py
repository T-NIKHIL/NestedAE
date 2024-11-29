""" Training autoencoder models using wandb sweeps """
import time
import os
import sys
import copy
from torch import load
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
from ae import AE
from wandb_api_key import api_key

def run_wandb_agent(run_dir, ae_save_dir, nn_params_dict, nn_train_params_dict, nn_datasets_dict, train_dataloader, val_dataloader, accelerator):
    """
    Runs the wandb agent for tuning neural network parameters.

    Args:
        run_dir (str): The run directory to store the ae models
        ae_save_dir (str): Directory to save the neural network model.
        nn_params_dict (dict): Dictionary containing the neural network parameters.
        nn_train_params_dict (dict): Dictionary containing the neural network training parameters.
        nn_datasets_dict (dict): Dictionary containing the neural network dataset parameters.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        accelerator (str): Accelerator to use for training.

    Returns:
        None
    """
    # To start a wandb run have to call wandb.init(). At any time there can only be one active run. To finish have to call run.finish()
    wandb.finish()
    run = wandb.init(job_type='training', resume=False, reinit=False, dir='../runs')
    new_nn_params_dict = copy.deepcopy(nn_params_dict)
    for submodule in nn_params_dict['submodules'].keys():
        for submodule_key in nn_params_dict['submodules'][submodule].keys():
            # Set the value chosen by the sweep
            if isinstance(new_nn_params_dict['submodules'][submodule][submodule_key], dict):
                if 'values' in list(new_nn_params_dict['submodules'][submodule][submodule_key].keys()):
                    new_nn_params_dict['submodules'][submodule][submodule_key] = wandb.config[f'{submodule}-{submodule_key}']
            # Get respective params from the encoder and set it for the submodule
            if new_nn_params_dict['submodules'][submodule][submodule_key] == 'mirror':
                new_nn_params_dict['submodules'][submodule][submodule_key] = wandb.config[f'encoder-{submodule_key}']
    
    # The directory to store each individual run
    model_sweep_dir = f'../runs/{run_dir}/{ae_save_dir}/ae_param_search/{run.name}'

    # Create a run directory under tune_nn_params
    if not os.path.exists(model_sweep_dir):
        os.makedirs(model_sweep_dir)

    # Send all print statements to file for debugging
    print_file_path = f'{model_sweep_dir}/train_out.txt'
    sys.stdout = open(print_file_path, "w", encoding='utf-8')

    # Build the nn model
    ae = AE(run_dir, ae_save_dir, run.name, new_nn_params_dict, nn_train_params_dict, nn_datasets_dict)
    ae.compile()
    print(' --> Model Compilation step complete.')

    callbacks = create_callback_object(nn_train_params_dict, model_sweep_dir)

    trainer = Trainer(max_epochs=nn_train_params_dict['epochs'],
                      accelerator=accelerator,
                      deterministic=True,
                      logger=WandbLogger(project=run_dir),
                      callbacks=callbacks,
                      enable_model_summary=False,
                      enable_progress_bar=False)
    
    trainer.fit(model=ae, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=None)
    wandb.finish()
    time.sleep(5)
    # # Move the files from wandb directory to model sweep directory
    # os.system(f'cp -r ../runs/wandb/*-{run.id}/* {model_sweep_dir}')
    # # Delete the run directory from wandb directory
    # os.system(f'rm -r {ae_save_dir}/ae_param_search/wandb/*-{run.id}')
    print(' --> EXIT.')

@click.command()
@click.option('--run_dir', prompt='run_dir',
            help='Specify the run directory.')
@click.option('--ae_save_dir', prompt='ae_save_dir',
               help='Specify the name of directory to save the neural network model.')
@click.option('--ae_idx', prompt='ae_idx',
            help='Specify the neural network to train.')
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
def run_wandb_sweep(run_dir, ae_save_dir, ae_idx, user_name, project_name, sweep_type, metric, goal, trials_in_sweep, accelerator):
    """
    Runs the hyperparameter tuning process for a neural network model by calling the wandb agent.

    Args:
        run_dir (str): The run directory to stor the ae models
        ae_save_dir (str): The directory to save the neural network model.
        nn (str): The neural network model to tune.
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
    os.environ['WANDB_DIR'] = '../runs'

    # Create a dicrectory to store wanb artifacts
    wandb_cache_path = f'../runs/wandb_cache'
    if not os.path.exists(wandb_cache_path):
        os.makedirs(wandb_cache_path)
        print(' --> Created wandb_cache directory.')

    # Create a directory to store wandb configs
    wandb_config_path = f'../runs/wandb_config'
    if not os.path.exists(wandb_config_path):
        os.makedirs(wandb_config_path)
        print(' --> Created wandb_config directory.')

    os.environ['WANDB_CONFIG_DIR'] = wandb_config_path
    os.environ['WANDB_CACHE_DIR'] = wandb_cache_path

    # Create a directory to store the hyperparameter runs
    train_path = f'..runs/{run_dir}/{ae_save_dir}/ae_param_search'
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        print(' --> Created ae_param_search directory.')

    ae_idx = int(ae_idx)
    nn_params_dict = list_of_nn_params_dict[ae_idx]
    nn_train_params_dict = list_of_nn_train_params_dict[ae_idx]
    nn_datasets_dict = list_of_nn_datasets_dict[ae_idx]

    # Set the  global random seed
    set_global_random_seed(nn_train_params_dict['global_seed'])

    if not os.path.exists(f'../runs/{run_dir}/{ae_save_dir}/datasets'):
        print(' --> Run in preprocess mode to create the datasets for training')
        sys.exit()

    # Load the train and validation datsets for the fold and create dataloaders
    train_dataset = load(f'../runs/{run_dir}/{ae_save_dir}/datasets/train_dataset.pt')
    train_dataloader = DataLoader(train_dataset,
                                    batch_size=nn_train_params_dict['batch_size'],
                                    shuffle=nn_train_params_dict['shuffle_data_between_epochs'],
                                    num_workers=0)
    val_dataset = load(f'../runs/{run_dir}/{ae_save_dir}/datasets/val_dataset.pt')
    val_dataloader = DataLoader(val_dataset,
                                batch_size=nn_train_params_dict['batch_size'],
                                shuffle=False,
                                num_workers=0)

    # Check if any submodules require optimization
    sweep=False
    for submodule in list(nn_params_dict['submodules'].keys()):
        if 'param_optimization' in list(nn_params_dict['submodules'][submodule].keys()):
            if nn_params_dict['submodules'][submodule]['param_optimization']:
                sweep=True
                break

    if sweep:
        params_to_tune = {}
        # Check for dictionary with 'values' key
        for submodule in list(nn_params_dict['submodules'].keys()):
            for submodule_key in list(nn_params_dict['submodules'][submodule].keys()):
                if isinstance(nn_params_dict['submodules'][submodule][submodule_key], dict):
                    if 'values' in list(nn_params_dict['submodules'][submodule][submodule_key].keys()):
                        params_to_tune[f'{submodule}-{submodule_key}'] = nn_params_dict['submodules'][submodule][submodule_key]
        # Define the sweep configuration
        sweep_config = {
            'name': list_of_nn_params_dict[ae_idx]['name'],
            'method': sweep_type,
            'metric': {
                'name': metric,
                'goal': goal
            },
            'parameters': params_to_tune
        }
        sweep_id = wandb.sweep(sweep_config)
        if int(trials_in_sweep) == -1:
            count = None
        else:
            count = int(trials_in_sweep)
        wandb.agent(sweep_id, lambda: run_wandb_agent(run_dir, ae_save_dir, nn_params_dict, nn_train_params_dict, nn_datasets_dict, train_dataloader, val_dataloader, accelerator), count=count)
    else:
        wandb.finish()
        # Do A single run
        run = wandb.init(job_type='training', resume=False, reinit=False, dir='../runs')
        # The directory to store each individual run
        model_sweep_dir = f'../runs/{run_dir}/{ae_save_dir}/ae_param_search/{run.name}'
        # Create a run directory under tune_nn_params
        if not os.path.exists(model_sweep_dir):
            os.makedirs(model_sweep_dir)
        # Send all print statements to file for debugging
        print_file_path = f'{model_sweep_dir}/train_out.txt'
        sys.stdout = open(print_file_path, "w", encoding='utf-8')
        # Build the nn model
        ae = AE(run_dir, ae_save_dir, run.name, nn_params_dict, nn_train_params_dict, nn_datasets_dict)
        ae.compile()
        print(' --> Model Compilation step complete.')
        callbacks = create_callback_object(nn_train_params_dict, model_sweep_dir)
        trainer = Trainer(max_epochs=nn_train_params_dict['epochs'],
                        accelerator=accelerator,
                        deterministic=True,
                        logger=WandbLogger(project=run_dir),
                        callbacks=callbacks,
                        enable_model_summary=False,
                        enable_progress_bar=False)
        trainer.fit(model=ae, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=None)
        wandb.finish()
        time.sleep(5)

if __name__ == '__main__':
    run_wandb_sweep()

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