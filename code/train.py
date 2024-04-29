""" Training script """

from nn.vanilla_ae import VanillaAE
from utils.dataset_utils import *
from utils.nn_utils import create_callback_object
from utils.custom_utils import set_global_random_seed, read_from_pickle, save_to_pickle

import json
import numpy as np
import click
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader
import torch
import sys

import logging
import os
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


@click.command()
@click.option('--run_dir', prompt='run_dir', help='Specify the run dir where the model is located.')
@click.option('--nn', prompt='nn', help='Specify neural network number used for making the prediction.')
@click.option('--accelerator', prompt='accelerator', help='Specify the type of acceleration to use.')
def train(run_dir, nn, accelerator):
    """ Training script"""

    nn_idx = int(nn) - 1

    run_dir = '../runs/' + run_dir

    # Read the ae params, train params and datasets
    list_of_nn_params_dict = read_from_pickle(
        'list_of_nn_params_dict.pkl', run_dir)
    list_of_nn_train_params_dict = read_from_pickle(
        'list_of_nn_train_params_dict.pkl', run_dir)
    list_of_nn_datasets_dict = read_from_pickle(
        'list_of_nn_datasets_dict.pkl', run_dir)

    nn_params_dict = list_of_nn_params_dict[nn_idx]
    nn_train_params_dict = list_of_nn_train_params_dict[nn_idx]
    nn_datasets_dict = list_of_nn_datasets_dict[nn_idx]

    global_seed = nn_train_params_dict['global_seed']

    nn_save_dir = run_dir + '/' + nn_params_dict['model_type']

    # Send all print statements to file for debugging
    print_file_path = nn_save_dir + '/' + 'train_out.txt'
    sys.stdout = open(print_file_path, "w", encoding='utf-8')

    print(f' --> User provided command line run_dir argument : {run_dir}')
    print(f' --> User provided command line nn idx argument : {nn_idx}')

    set_global_random_seed(global_seed)

    print(f' --> Setting global random seed {global_seed}.')

    print(f' --> Running on {accelerator}.')

    print(f' --> Number of threads : {torch.get_num_threads()}')
    print(
        f' --> Number of interop threads : {torch.get_num_interop_threads()}')

    print(' --> PyTorch configurations')
    # torch.__config__.show()
    # torch.__config__.parallel_info()

    # Save user provided nn dictionaries to pickle
    save_to_pickle(list_of_nn_datasets_dict,
                   'list_of_nn_datasets_dict.pkl', run_dir)
    save_to_pickle(list_of_nn_params_dict,
                   'list_of_nn_params_dict.pkl', run_dir)
    save_to_pickle(list_of_nn_train_params_dict,
                   'list_of_nn_train_params_dict.pkl', run_dir)
    print(' --> Saved user provided dictionaries to pickle.')

    # Save the history of all different models created in the run directory.
    with open(run_dir + '/' + 'run_summary.txt', 'a', encoding='utf-8') as file:
        for model_num in enumerate(list_of_nn_params_dict):
            file.write(f'--NN params dict (Model {model_num})--' + '\n')
            file.write(json.dumps(
                list_of_nn_params_dict[model_num], indent=4) + '\n')
            file.write('\n')

            file.write(
                f'--NN train params dict (Model {model_num})--' + '\n')
            file.write(json.dumps(
                list_of_nn_train_params_dict[model_num], indent=4) + '\n')
            file.write('\n')

            file.write(
                f'--NN dataset dict (Model {model_num})--' + '\n')
            file.write(json.dumps(
                list_of_nn_datasets_dict[model_num], indent=4) + '\n')
            file.write('\n')
    print(' --> Saved user provided dictionaries to run_summary.txt')

    # Load the pytorch datasets
    dataset_save_dir = nn_save_dir + '/datasets'

    # If num workers is 0 then main process will be used for loading the dada
    # prefetch factor deermines number of batches prefetched across all workers

    train_dataset_path = dataset_save_dir + '/train_dataset.pt'
    train_dataset = torch.load(train_dataset_path)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=nn_train_params_dict['batch_size'],
                                  shuffle=nn_train_params_dict['shuffle_data_between_epochs'],
                                  num_workers=0)

    val_dataset_path = dataset_save_dir + '/val_dataset.pt'
    val_dataset = torch.load(val_dataset_path)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=nn_train_params_dict['batch_size'],
                                shuffle=False,
                                num_workers=0)
    
    ae = VanillaAE(nn_save_dir,
                   nn_params_dict,
                   nn_train_params_dict,
                   nn_datasets_dict)
    
    # Model Compile Step
    ae.compile()
    print(' --> Model Compilation step complete.')

    # Define Pytorch Trainer
    # Extract parameters needed for the training loop
    epochs = nn_train_params_dict['epochs']
    loggers = []
    # Log the model training to Tensorboard and CSV
    logs_dir = nn_save_dir + '/logs'
    if os.path.exists(logs_dir) is False:
        os.mkdir(logs_dir)
    csv_logger = CSVLogger(logs_dir, name='csv_logs')
    loggers.append(csv_logger)
    # accelearator set to 'auto' for automatic detection of which system to train on
    callbacks = create_callback_object(nn_train_params_dict, nn_save_dir)
    trainer = Trainer(max_epochs=epochs,
                      accelerator=accelerator,
                      deterministic=True,
                      logger=loggers,
                      callbacks=callbacks)

    # Specify last to resume training from last checkpoint
    trainer.fit(model=ae,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=None)

    # Code to verify nn checkpoint
    '''
    # Create a new ae module
    new_ae = create_ae_module(nn_save_dir,
                              nn_params_dict,
                              nn_train_params_dict,
                              nn_datasets_dict)

    # Load the weights from the latest checkpoint
    chpt_path = nn_save_dir + '/checkpoints/last.ckpt'
    loaded_ae = new_ae.load_from_checkpoint(chpt_path)

    submodule_outputs_from_loaded = loaded_ae(loaded_ae.all_samples)

    latents_from_loaded = submodule_outputs_from_loaded['latent']

    submodule_outputs = ae(ae.all_samples)

    latents = submodule_outputs['latent']

    if torch.equal(latents_from_loaded, latents):
        print('Loaded latents equal to latents produced from trained ae.')
    else:
        raise Exception('loaded latents not equal to latents from trained ae. Check checkpoint file. ')
    '''

    print(' --> EXIT.')


if __name__ == '__main__':
    train()
