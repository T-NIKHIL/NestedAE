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
be_data_train = pd.read_csv('be_data_train.csv')
be_data_val = pd.read_csv('be_data_val.csv')

be_true_train = be_data_train['True Binding Energy'].values
be_pred_train = be_data_train['Predicted Binding Energy'].values
be_true_val = be_data_val['True Binding Energy'].values
be_pred_val = be_data_val['Predicted Binding Energy'].values

be_true = np.concatenate((be_true_train, be_true_val), axis=0)
be_pred = np.concatenate((be_pred_train, be_pred_val), axis=0)

# Scatter plot true and predicted bandgaps
fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))
ax.scatter(be_true_train, be_pred_train, alpha=1.0, s=5, label='Train', c='blue')
ax.scatter(be_true_val, be_pred_val, alpha=1.0, s=5, label='Validation', c='orange')
ax.set_xlabel(r'True Binding Energy (kJ/mol)')
ax.set_ylabel(r'Predicted Binding Energy (kJ/mol)')
ax.plot([be_true.min(), be_true.max()], [be_true.min(), be_true.max()], 'r--', lw=1)
ax.set_xlim(be_true.min()-1, be_true.max()+1)
ax.set_ylim(be_true.min()-1, be_true.max()+1)
# Insert text box to the lower right hand corner
# Remove frame for text box
ax.text(0.42, 0.13,
           f'Train MAE: {np.mean(np.abs(be_pred_train - be_true_train)):.2f} kJ/mol \n \
             Val. MAE: {np.mean(np.abs(be_pred_val - be_true_val)):.2f} kJ/mol',
           transform=ax.transAxes,
           fontsize=8,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
ax.legend(frameon=False, fontsize=8)
ax.set_aspect('equal', 'box')
ax.set_xticks(np.arange(5, 45, 5))
ax.set_yticks(np.arange(5, 45, 5))
plt.tight_layout()
plt.savefig('hd2_mhp_binding_energy_true_vs_pred.pdf', bbox_inches='tight', dpi=300)