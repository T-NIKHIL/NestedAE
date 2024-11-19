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
@click.option('--run_dir', prompt='run_dir', 
              help='The run directory contains all the NestedAE models trained on the multiscale dataset.')
@click.option('--nn_save_dir', prompt='nn_save_dir', 
              help='Specify name of the directory to store the model.')
@click.option('--nn', prompt='nn', help='Specify neural network number used for making the prediction.')
@click.option('--accelerator', prompt='accelerator', help='Specify the type of acceleration to use.')
def train(run_dir, nn_save_dir, nn, accelerator):
    """ Training script"""

    nn_save_dir_path = f'../runs/{run_dir}/{nn_save_dir}'

    # Read the ae params, train params and datasets
    list_of_nn_params_dict = read_from_pickle(
        'list_of_nn_params_dict.pkl', nn_save_dir_path)
    list_of_nn_train_params_dict = read_from_pickle(
        'list_of_nn_train_params_dict.pkl', nn_save_dir_path)
    list_of_nn_datasets_dict = read_from_pickle(
        'list_of_nn_datasets_dict.pkl', nn_save_dir_path)
    
    nn_idx = int(nn) - 1

    nn_params_dict = list_of_nn_params_dict[nn_idx]
    nn_train_params_dict = list_of_nn_train_params_dict[nn_idx]
    nn_datasets_dict = list_of_nn_datasets_dict[nn_idx]

    global_seed = nn_train_params_dict['global_seed']

    # Send all print statements to file for debugging
    print_file_path = nn_save_dir_path + '/' + 'train_out.txt'
    sys.stdout = open(print_file_path, "w", encoding='utf-8')

    print(f' --> User provided command line run_dir argument : {run_dir}')
    print(f' --> User provided command line nn idx argument : {nn_idx}')

    set_global_random_seed(global_seed)

    print(f' --> Setting global random seed {global_seed}.')
    print(f' --> Running on {accelerator}.')

    # Load the pytorch datasets
    dataset_save_dir = nn_save_dir_path + '/datasets'

    # If num workers is 0 then main process will be used for loading the dada
    # prefetch factor deermines number of batches prefetched across all workers

    train_dataset_path = dataset_save_dir + '/train_dataset.pt'
    train_dataset = torch.load(train_dataset_path)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=nn_train_params_dict['batch_size'],
                                  shuffle=nn_train_params_dict['shuffle_data_between_epochs'])

    val_dataset_path = dataset_save_dir + '/val_dataset.pt'
    val_dataset = torch.load(val_dataset_path)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=nn_train_params_dict['batch_size'],
                                shuffle=False)
    
    ae = VanillaAE(nn_save_dir_path,
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
    logs_dir = nn_save_dir_path + '/logs'
    if os.path.exists(logs_dir) is False:
        os.mkdir(logs_dir)
    csv_logger = CSVLogger(logs_dir, name='csv_logs')
    loggers.append(csv_logger)
    # accelearator set to 'auto' for automatic detection of which system to train on
    callbacks = create_callback_object(nn_train_params_dict, nn_save_dir_path)
    trainer = Trainer(max_epochs=epochs,
                      accelerator=accelerator,
                      deterministic=True,
                      logger=loggers,
                      callbacks=callbacks,
                      enable_model_summary=False,
                      enable_progress_bar=False,
                      log_every_n_steps=nn_train_params_dict['batch_size'])

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
