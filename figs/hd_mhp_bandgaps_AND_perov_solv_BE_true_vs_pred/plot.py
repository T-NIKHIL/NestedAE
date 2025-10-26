import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Plotting parameters
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

# Load the data from csv file
bandgaps_data_train = pd.read_csv('bandgaps_data_train.csv')
bandgaps_data_val = pd.read_csv('bandgaps_data_val.csv')
be_data_train = pd.read_csv('be_data_train.csv')
be_data_val = pd.read_csv('be_data_val.csv')

bandgaps_true_train = bandgaps_data_train['True Bandgaps'].values
bandgaps_pred_train = bandgaps_data_train['Predicted Bandgaps'].values
bandgaps_true_val = bandgaps_data_val['True Bandgaps'].values
bandgaps_pred_val = bandgaps_data_val['Predicted Bandgaps'].values

bandgaps_true = np.concatenate((bandgaps_true_train, bandgaps_true_val), axis=0)
bandgaps_pred = np.concatenate((bandgaps_pred_train, bandgaps_pred_val), axis=0)

be_true_train = be_data_train['True Binding Energy'].values
be_pred_train = be_data_train['Predicted Binding Energy'].values
be_true_val = be_data_val['True Binding Energy'].values
be_pred_val = be_data_val['Predicted Binding Energy'].values

be_true = np.concatenate((be_true_train, be_true_val), axis=0)
be_pred = np.concatenate((be_pred_train, be_pred_val), axis=0)

# Scatter plot true and predicted bandgaps
fig, ax = plt.subplots(1, 2, figsize=(6.0, 3.0))
ax[0].scatter(bandgaps_true_train, bandgaps_pred_train, alpha=1.0, s=5, c='blue', label='Train')
ax[0].scatter(bandgaps_true_val, bandgaps_pred_val, alpha=1.0, s=5, c='orange', label='Validation')
ax[0].set_xlabel(r'True Bandgap (eV)')
ax[0].set_ylabel(r'Predicted Bandgap (eV)')
ax[0].plot([bandgaps_true.min(), bandgaps_true.max()], [bandgaps_true.min(), bandgaps_true.max()], 'r--', lw=1)
ax[0].set_xlim(bandgaps_true.min()-0.1, bandgaps_true.max()+0.1)
ax[0].set_ylim(bandgaps_true.min()-0.1, bandgaps_true.max()+0.1)
# Insert text box to the lower right hand corner
# Remove frame for text box
ax[0].text(0.52, 0.12, 
           f'Train MAE: {np.mean(np.abs(bandgaps_pred_train - bandgaps_true_train)):.2f} eV \n \
             Val. MAE: {np.mean(np.abs(bandgaps_pred_val - bandgaps_true_val)):.2f} eV', 
           transform=ax[0].transAxes, 
           fontsize=8,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
ax[0].legend(frameon=False, fontsize=8)
ax[0].set_aspect('equal', 'box')
ax[0].set_xticks(np.arange(1.4, 2.6, 0.2))
ax[0].set_yticks(np.arange(1.4, 2.6, 0.2))

ax[1].scatter(be_true_train, be_pred_train, alpha=1.0, s=5, c='blue', label='Train')
ax[1].scatter(be_true_val, be_pred_val, alpha=1.0, s=5, c='orange', label='Validation')
ax[1].set_xlabel(r'True Binding Energy (kJ/mol)')
ax[1].set_ylabel(r'Predicted Binding Energy (kJ/mol)')
ax[1].plot([be_true.min(), be_true.max()], [be_true.min(), be_true.max()], 'r--', lw=1)
ax[1].set_xlim(be_true.min()-1, be_true.max()+1)
ax[1].set_ylim(be_true.min()-1, be_true.max()+1)
# Insert text box to the lower right hand corner
# Remove frame for text box
ax[1].text(0.42, 0.13, 
           f'Train MAE: {np.mean(np.abs(be_pred_train - be_true_train)):.2f} kJ/mol \n \
             Val. MAE: {np.mean(np.abs(be_pred_val - be_true_val)):.2f} kJ/mol', 
           transform=ax[1].transAxes, 
           fontsize=8,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
ax[1].legend(frameon=False, fontsize=8)
ax[1].set_aspect('equal', 'box')
ax[1].set_xticks(np.arange(5, 45, 5))
ax[1].set_yticks(np.arange(5, 45, 5))
plt.tight_layout()
plt.savefig('ae1_mhp_bandgaps_AND_perov_solv_BE_true_vs_pred.pdf', bbox_inches='tight', dpi=300)
