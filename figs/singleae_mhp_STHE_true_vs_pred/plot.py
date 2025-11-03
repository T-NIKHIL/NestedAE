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
  # SingleAE Parameters
  #################################################################

  model_dir = '../../runs/singleae_perov_STHE/fold0_21D'
  latent_dim = 21
  fold_num = 0
  module_params = {'name':'AE1', 
                      'modules':{

                          'encoder':{
                              'input_dim':26,
                              'output_dim':latent_dim, 
                              'hidden_dim':50, 
                              'hidden_layers':1, 
                              'hidden_activation':None, 
                              'output_activation':torch.nn.Tanh(), 
                              'layer_kernel_init':'xavier_normal', 
                              'layer_bias_init':'zeros'},

                          'STHE_predictor':{
                              'input_dim':latent_dim,
                              'output_dim':1,
                              'hidden_dim':50,
                              'hidden_layers':1,
                              'hidden_activation':torch.nn.ReLU(),
                              'output_activation':torch.nn.ReLU(),
                              'layer_kernel_init':'xavier_normal',
                              'layer_bias_init':'zeros'},

                          'A_predictor':{
                            'input_dim':latent_dim,
                            'output_dim':4,
                            'hidden_dim':50,
                            'hidden_layers':1,
                            'hidden_activation':torch.nn.ReLU(),
                            'output_activation':None, # Logits
                            'layer_kernel_init':'xavier_normal',
                            'layer_bias_init':'zeros'},
                            

                        'B_predictor':{
                            'input_dim':latent_dim,
                            'output_dim':6,
                            'hidden_dim':50,
                            'hidden_layers':1,
                            'hidden_activation':torch.nn.ReLU(),
                            'output_activation':None, # Logits
                            'layer_kernel_init':'xavier_normal',
                            'layer_bias_init':'zeros'},

                        'X_predictor':{
                            'input_dim':latent_dim,
                            'output_dim':3,
                            'hidden_dim':50,
                            'hidden_layers':1,
                            'hidden_activation':torch.nn.ReLU(),
                            'output_activation':None, # Logits
                            'layer_kernel_init':'xavier_normal',
                            'layer_bias_init':'zeros'},  

                          'phase_predictor':{
                              'input_dim':latent_dim,
                              'output_dim':4,
                              'hidden_dim':50,
                              'hidden_layers':1,
                              'hidden_activation':torch.nn.ReLU(),
                              'output_activation':None,
                              'layer_kernel_init':'xavier_normal',
                              'layer_bias_init':'zeros'}, 

                          'decoder':{
                              'input_dim':latent_dim,
                              'output_dim':26,
                              'hidden_dim':50,
                              'hidden_layers':1,
                              'hidden_activation':torch.nn.ReLU(),
                              'output_activation':None,
                              'layer_kernel_init':'xavier_normal',
                              'layer_bias_init':'zeros'
                          }

                      }}

  #################################################################
  # Dataset Parameters
  #################################################################

  train_dataset_loc = '../../datasets/MHP_for_water_splitting/dataset.csv'
  test_dataset_loc = '../../datasets/MHP_for_water_splitting/test_dataset_mixed_A.csv'

  latent_col_names = []
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
                  'X_AN',
                  'A_En_mull',
                  'B_En_mull',
                  'X_En_mull',
                  'x(S)',
                  't',
                  'o',
                  'tao',
                  'CBM',
                  'VBM',
                  'n_abs(%)',
                  'n_cu(%)']
  target = ['n_STH(%)']
  target_A_ion = ['Rb', 'Cs', 'MA', 'FA']
  target_B_ion = ['Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb']
  target_X_ion = ['Cl', 'Br', 'I']
  target_phase = ['Cubic', 'Tetra', 'Ortho', 'Hex']

  standardize_descs = True

  # #################################################################
  # --------------------- END OF USER INPUT ---------------------
  # #################################################################

  x_dataframe_train = pd.read_csv(train_dataset_loc)[descriptors + latent_col_names]
  x_dataframe_test = pd.read_csv(test_dataset_loc)[descriptors + latent_col_names]
  y_dataframe_test = pd.read_csv(test_dataset_loc)[target]

  if standardize_descs:
      desc_means_dict = {}
      desc_std_devs = {}
      for desc in x_dataframe_train.columns.tolist():
          mean = x_dataframe_train[desc].mean()
          desc_means_dict[desc] = mean
          std_dev = x_dataframe_train[desc].std()
          desc_std_devs[desc] = std_dev
          x_dataframe_test[desc] = (x_dataframe_test[desc] - mean) / std_dev
      print('(INFO) Descriptors standardized.')
  else:
      print('(INFO) Descriptors not standardized.')

  test_dataset_torch = torch.from_numpy(x_dataframe_test.to_numpy(dtype=np.float32))
  sthe_true_test = y_dataframe_test.to_numpy(dtype=np.float32).squeeze()

  # Load the single autoencoder model
  loaded_model = AE(module_params)
  loaded_model.load_state_dict(torch.load(model_dir))

  loaded_model.eval()
  with torch.no_grad():
    for i, layer in enumerate(loaded_model.ae_modules['encoder']):
      if i == 0:
        latents = layer(test_dataset_torch)
      else:
        latents = layer(latents)
    
    for i, layer in enumerate(loaded_model.ae_modules['STHE_predictor']):
      if i == 0:
        sthe_pred_test = layer(latents)
      else:
        sthe_pred_test = layer(sthe_pred_test)

  # loaded_model.eval()
  # with torch.no_grad():
  #   for i, layer in enumerate(loaded_model.ae_modules['encoder']):
  #     if i == 0:
  #       latents = layer(val_dataset_torch)
  #     else:
  #       latents = layer(latents)
    
  #   for i, layer in enumerate(loaded_model.ae_modules['STHE_predictor']):
  #     if i == 0:
  #       sthe_pred_val = layer(latents)
  #     else:
  #       sthe_pred_val = layer(sthe_pred_val)

  # Scatter plot true and predicted bandgaps
  fig, ax = plt.subplots(figsize=(3.0, 3.0))
  # ax.scatter(sthe_true_test, sthe_pred_test.detach().cpu().numpy(), alpha=1.0, s=5, c='blue', label='Train')
  ax.scatter(sthe_true_test, sthe_pred_test.detach().cpu().numpy(), alpha=1.0, s=5, c='orange', label='Validation')
  ax.set_xlabel(r'True STHE \( \% \)')
  ax.set_ylabel(r'Predicted STHE \( \% \)')
  ax.plot([sthe_true_test.min()-5, sthe_true_test.max()+5], [sthe_true_test.min()-5, sthe_true_test.max()+5], 'r--', lw=1)
  ax.set_xlim(sthe_true_test.min()-5, sthe_true_test.max()+5)
  ax.set_ylim(sthe_true_test.min()-5, sthe_true_test.max()+5)
  # ax.grid()
  # Insert text box to the lower right hand corner
  # ax.text(0.52, 0.12,
  #           f'Train MAE: {np.mean(np.abs(sthe_pred_train.detach().cpu().numpy().squeeze() - sthe_true_train)):.3f} \n \
  #             Val. MAE: {np.mean(np.abs(sthe_pred_val.detach().cpu().numpy().squeeze() - sthe_true_val)):.3f}',
  #           transform=ax.transAxes,
  #           fontsize=8,
  #           verticalalignment='top',
  #           bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='none'))
  ax.text(0.52, 0.12,
          f'Test MAE: {np.mean(np.abs(sthe_pred_test.detach().cpu().numpy().squeeze() - sthe_true_test)):.3f} \n \
            Test RMSE: {np.sqrt(np.mean((sthe_pred_test.detach().cpu().numpy().squeeze() - sthe_true_test)**2)):.3f}',
          transform=ax.transAxes,
          fontsize=8,
          verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='none'))
  # ax.legend(frameon=False, fontsize=8)
  ax.set_aspect('equal', 'box')
  ax.set_xticks(np.arange(0, 35, 5))
  ax.set_yticks(np.arange(0, 35, 5))
  plt.tight_layout()
  plt.savefig('singleae_mhp_sthe_true_vs_pred.pdf', bbox_inches='tight', dpi=300)