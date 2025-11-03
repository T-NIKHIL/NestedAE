import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ks_2samp

from model import AE, Arctanh

atanh_act_fn = Arctanh()

if __name__ == "__main__":

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

  model_dir = '../../runs/nestedae_ae2_perov_STHE/fold2_10D'
  latent_dim = 10
  fold_num = 2
  module_params = {'name':'AE1', 
                      'modules':{

                          'encoder':{
                              'input_dim':14,
                              'output_dim':latent_dim, 
                              'hidden_dim':25, 
                              'hidden_layers':1, 
                              'hidden_activation':None, 
                              'output_activation':torch.nn.Tanh(), 
                              'layer_kernel_init':'xavier_normal', 
                              'layer_bias_init':'zeros'},

                          'STHE_predictor':{
                              'input_dim':latent_dim,
                              'output_dim':1,
                              'hidden_dim':25,
                              'hidden_layers':1,
                              'hidden_activation':None,
                              'output_activation':torch.nn.ReLU(),
                              'layer_kernel_init':'xavier_normal',
                              'layer_bias_init':'zeros'},

                          'phase_predictor':{
                              'input_dim':latent_dim,
                              'output_dim':4,
                              'hidden_dim':25,
                              'hidden_layers':1,
                              'hidden_activation':torch.nn.ReLU(),
                              'output_activation':None,
                              'layer_kernel_init':'xavier_normal',
                              'layer_bias_init':'zeros'},    
                          
                          'latents_predictor':{
                              'input_dim':latent_dim,
                              'output_dim':7,
                              'hidden_dim':25,
                              'hidden_layers':1,
                              'hidden_activation':None,
                              'output_activation':None,
                              'layer_kernel_init':'xavier_normal',
                              'layer_bias_init':'zeros'},   

                          'decoder':{
                              'input_dim':latent_dim,
                              'output_dim':7,
                              'hidden_dim':25,
                              'hidden_layers':1,
                              'hidden_activation':None,
                              'output_activation':None,
                              'layer_kernel_init':'xavier_normal',
                              'layer_bias_init':'zeros'
                          }

                      }}

  #################################################################
  # Dataset Parameters
  #################################################################

  dataset_loc = '../../datasets/MHP_for_water_splitting/dataset_with_ae1_latents.csv'
  latent_col_names = ['l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6']
  descriptors = ['t',
                'o',
                'tao',
                'CBM',
                'VBM',
                'n_abs(%)',
                'n_cu(%)']
  target = ['n_STH(%)']
  target_phase = ['Cubic', 'Tetra', 'Ortho', 'Hex']

  standardize_descs = True
  defined_qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  train_split = 0.9

  # #################################################################
  # --------------------- END OF USER INPUT ---------------------
  # #################################################################

  x_dataframe = pd.read_csv(dataset_loc)[descriptors + latent_col_names]
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
  train_dataset_torch = torch.from_numpy(train_dataset)
  val_dataset_torch = torch.from_numpy(val_dataset)

  sthe_true_train = y_dataframe.to_numpy(dtype=np.float32)[train_idxs[fold_num]].squeeze()
  sthe_true_val = y_dataframe.to_numpy(dtype=np.float32)[val_idxs[fold_num]].squeeze()

  # Load the nested autoencoder model
  loaded_model = AE(module_params)
  loaded_model.load_state_dict(torch.load(model_dir))

  loaded_model.eval()
  with torch.no_grad():
    for i, layer in enumerate(loaded_model.ae_modules['encoder']):
      if i == 0:
        latents = layer(train_dataset_torch)
      else:
        latents = layer(latents)
    
    for i, layer in enumerate(loaded_model.ae_modules['STHE_predictor']):
      if i == 0:
        sthe_pred_train = layer(latents)
      else:
        sthe_pred_train = layer(sthe_pred_train)

  loaded_model.eval()
  with torch.no_grad():
    for i, layer in enumerate(loaded_model.ae_modules['encoder']):
      if i == 0:
        latents = layer(val_dataset_torch)
      else:
        latents = layer(latents)
    
    for i, layer in enumerate(loaded_model.ae_modules['STHE_predictor']):
      if i == 0:
        sthe_pred_val = layer(latents)
      else:
        sthe_pred_val = layer(sthe_pred_val)

  sthe_true = np.concatenate((sthe_true_train, sthe_true_val), axis=0)
  sthe_pred = np.concatenate((sthe_pred_train.detach().cpu().numpy(), sthe_pred_val.detach().cpu().numpy()), axis=0)

  # Scatter plot true and predicted bandgaps
  fig, ax = plt.subplots(figsize=(3.0, 3.0))
  ax.scatter(sthe_true_train, sthe_pred_train.detach().cpu().numpy(), alpha=1.0, s=5, c='blue', label='Train')
  ax.scatter(sthe_true_val, sthe_pred_val.detach().cpu().numpy(), alpha=1.0, s=5, c='orange', label='Validation')
  ax.set_xlabel(r'True STHE \( \% \)')
  ax.set_ylabel(r'Predicted STHE \( \% \)')
  ax.plot([sthe_true.min()-5, sthe_true.max()+5], [sthe_true.min()-5, sthe_true.max()+5], 'r--', lw=1)
  ax.set_xlim(sthe_true.min()-5, sthe_true.max()+5)
  ax.set_ylim(sthe_true.min()-5, sthe_true.max()+5)
  # ax.grid()
  # Insert text box to the lower right hand corner
  ax.text(0.52, 0.12,
            f'Train MAE: {np.mean(np.abs(sthe_pred_train.detach().cpu().numpy().squeeze() - sthe_true_train)):.3f} \n \
              Val. MAE: {np.mean(np.abs(sthe_pred_val.detach().cpu().numpy().squeeze() - sthe_true_val)):.3f}',
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='none'))
  ax.legend(frameon=False, fontsize=8)
  ax.set_aspect('equal', 'box')
  ax.set_xticks(np.arange(0, 35, 5))
  ax.set_yticks(np.arange(0, 35, 5))
  plt.tight_layout()
  plt.savefig('ae2_mhp_sthe_true_vs_pred.pdf', bbox_inches='tight', dpi=300)