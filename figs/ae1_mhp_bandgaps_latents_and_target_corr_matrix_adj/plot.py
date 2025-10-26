import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ks_2samp

from model import AE, Arctanh

atanh_act_fn = Arctanh()

if __name__ == '__main__':

    # #################################################################
    # --------------------- START OF USER INPUT ---------------------
    # #################################################################

    random_state = 42

    #################################################################
    # Plotting parameters
    #################################################################
    plt.rcParams.update({
    "text.usetex":True,
    "font.family":"sans-serif",
    "font.serif":["Computer Modern Roman"]})
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 10     
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.minor.width'] = 1

    #################################################################
    # Autoencoder 1 Parameters
    #################################################################

    model_dir = '../../runs/ae1_bandgaps_THEN_perov_solv_BE/fold2_soft_constraints_11D'
    latent_dim = 11
    fold_num = 2
    module_params = {'name':'AE1',
                            'modules':{
                                
                                'encoder':{
                                    'input_dim':15,
                                    'output_dim':latent_dim, 
                                    'hidden_dim':25,
                                    'hidden_layers':1, 
                                    'hidden_activation':None, 
                                    'output_activation':torch.nn.Tanh(), 
                                    'layer_kernel_init':'xavier_normal', 
                                    'layer_bias_init':'zeros'},


                                'bandgaps_predictor':{
                                    'input_dim':latent_dim,
                                    'output_dim':1,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':torch.nn.ReLU(),
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'},

                                'A_predictor':{
                                    'input_dim':latent_dim,
                                    'output_dim':5,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':None,
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'
                                },

                                'B_predictor':{
                                    'input_dim':latent_dim,
                                    'output_dim':6,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':None,
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'
                                },

                                'X_predictor':{
                                    'input_dim':latent_dim,
                                    'output_dim':3,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':None,
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'
                                },    

                                'decoder':{
                                    'input_dim':latent_dim,
                                    'output_dim':15,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':None,
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'},

                    }}

    #################################################################
    # Dataset Parameters
    #################################################################

    dataset_loc = '../../datasets/MHP_bandgaps_AND_perov_solvent_BE/nestedae_dataset/perov_bandgaps_PBE_arun_reduced.csv'
    descriptors = ['A_IONRAD',
                   'A_MASS',
                   'A_DPM',
                   'B_IONRAD',
                   'B_MASS',
                   'B_EA',
                   'B_IE',
                   'B_En',
                   'B_AN', 
                   'X_IONRAD',
                   'X_MASS',
                   'X_EA',
                   'X_IE',
                   'X_En',
                   'X_AN']
    target = ['Gap']

    standardize_descs = True
    defined_qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    train_split = 0.9

    #################################################################
    # Script Parameters
    #################################################################

    load_data = False
    train_or_test = 'test'  # 'train' or 'test'

    # #################################################################
    # --------------------- END OF USER INPUT ---------------------
    # #################################################################

    x_dataframe = pd.read_csv(dataset_loc)[descriptors]
    y_dataframe = pd.read_csv(dataset_loc)[target]

    if standardize_descs:
        desc_means_dict = {}
        desc_std_devs = {}
        for desc in x_dataframe.columns.tolist():
            mean = x_dataframe[desc].mean()
            desc_means_dict[desc] = mean
            std_dev = x_dataframe[desc].std()
            desc_std_devs[desc] = std_dev
            x_dataframe[desc] = (x_dataframe[desc] - mean) / std_dev
        print('(INFO) Descriptors standardized.')
    else:
        print('(INFO) Descriptors not standardized.')
    
    print('(INFO) Using a stratified k fold split strategy.')
    train_idxs = []
    val_idxs = []
    y = y_dataframe[target].to_numpy(dtype=np.float32)
    skf = StratifiedKFold(n_splits=int(1/(1 - train_split)), shuffle=True, random_state=random_state)
    y_binned = np.digitize(y, np.quantile(y, defined_qs))
    for fold, (train_idx, val_idx) in enumerate(skf.split(x_dataframe.to_numpy(dtype=np.float32), y_binned)):
        train_idxs.append(train_idx)
        val_idxs.append(val_idx)
        ks_stat, p_val = ks_2samp(y[train_idx], y[val_idx])
        print(f'(INFO) Fold {fold} ks-stat for target: {np.round(ks_stat, 3)}, p-value: {np.round(p_val, 3)}')

    train_dataset = x_dataframe.to_numpy(dtype=np.float32)[train_idxs[fold_num]]
    val_dataset = x_dataframe.to_numpy(dtype=np.float32)[val_idxs[fold_num]]
    train_torch = torch.from_numpy(train_dataset)
    val_torch = torch.from_numpy(val_dataset)

    train_y = torch.from_numpy(y_dataframe.to_numpy(dtype=np.float32)[train_idxs[fold_num]])
    val_y = torch.from_numpy(y_dataframe.to_numpy(dtype=np.float32)[val_idxs[fold_num]])

    # Load the nested autoencoder model
    loaded_model = AE(module_params)
    loaded_model.load_state_dict(torch.load(model_dir))

    if train_or_test == 'train':
        loaded_model.eval()
        with torch.no_grad():
            for i, layer in enumerate(loaded_model.ae_modules['encoder']):
                if i == 0:
                    latents = layer(train_torch)
                else:
                    latents = layer(latents)
        latents_and_target = torch.cat((latents, train_y.reshape(-1, 1)), dim=1)
        std_err_PCC = 1/math.sqrt(len(train_y) - 3)

    if train_or_test == 'test':
        loaded_model.eval()
        with torch.no_grad():
            for i, layer in enumerate(loaded_model.ae_modules['encoder']):
                if i == 0:
                    latents = layer(val_torch)
                else:
                    latents = layer(latents)
        latents_and_target = torch.cat((latents, val_y.reshape(-1, 1)), dim=1)
        std_err_PCC = 1/math.sqrt(len(val_y) - 3)

    latents_and_target_corr_matrix = np.abs(np.round(np.corrcoef(x=latents_and_target.detach().numpy(), rowvar=False), 3))
    latents_and_target_corr_matrix_adj = (latents_and_target_corr_matrix - std_err_PCC)/(1 - std_err_PCC)
    # Clip less than 0 to 0
    latents_and_target_corr_matrix_adj[latents_and_target_corr_matrix_adj < 0] = 0
    # Print the adjusted correlation matrix
    print('latents_and_target_corr_matrix_adj:', latents_and_target_corr_matrix_adj)
    # Save the correlation matrix as csv
    np.savetxt('data.csv', latents_and_target_corr_matrix_adj, delimiter=',')

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    pcc_plot = ax.matshow(latents_and_target_corr_matrix_adj, cmap='Oranges', vmin=0, vmax=1)
    labels = [r'$l_0$', r'$l_1$', r'$l_2$', r'$l_3$', r'$l_4$', r'$l_5$', r'$l_6$', r'$l_7$', r'$l_8$', r'$l_9$', r'$l_{10}$','bg']
    ax.set_yticks(ticks=np.arange(len(labels)), labels=labels, rotation=0)
    ax.set_xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90)
    cbar = plt.colorbar(pcc_plot, ax=ax, shrink=0.72, pad=0.02)
    # cbar.ax.tick_params(labelsize=8)

    cbar.set_label(r'$\tilde{\rho}$', rotation=270, labelpad=10)
    plt.tight_layout()
    plt.savefig('ae1_mhp_latents_and_target_corr_matrix_adj.pdf', format='pdf', dpi=300, bbox_inches='tight')