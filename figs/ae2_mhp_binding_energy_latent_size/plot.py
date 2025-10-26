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

#
# NestedAE
#

# Load the data from csv file
latent_dims =            [17,   16,   15,   14,   13,   12,   11,   10,   9,    8,    7,    6,    5,    4,    3,    2]

train_be_pred_mean =     [0.58, 0.52, 0.51, 0.57, 0.50, 0.55, 0.59, 0.76, 0.67, 0.63, 0.66, 0.76, 0.82, 0.94, 1.08, 1.09]
train_be_pred_std =      [0.15, 0.06, 0.04, 0.05, 0.02, 0.03, 0.06, 0.26, 0.15, 0.04, 0.04, 0.08, 0.04, 0.06, 0.23, 0.22]

train_design_pred_mean = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.01, 0.01, 0.02, 0.02, 0.25, 0.36, 0.47]
train_design_pred_std =  [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.05, 0.00, 0.00, 0.02, 0.00, 0.10, 0.19, 0.20]

train_latent_pred_mean = [0.06, 0.04, 0.03, 0.04, 0.03, 0.02, 0.04, 0.09, 0.04, 0.02, 0.03, 0.07, 0.13, 0.18, 0.24, 0.54]
train_latent_pred_std =  [0.03, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.12, 0.03, 0.00, 0.01, 0.06, 0.08, 0.11, 0.13, 0.12]

train_x_pred_mean =      [0.03, 0.02, 0.02, 0.03, 0.02, 0.02, 0.02, 0.05, 0.06, 0.02, 0.02, 0.04, 0.07, 0.33, 0.39, 0.38]
train_x_pred_std =       [0.01, 0.01, 0.01, 0.01, 0.00, 0.01, 0.01, 0.06, 0.10, 0.00, 0.00, 0.03, 0.05, 0.33, 0.06, 0.06]

val_be_pred_mean  =      [1.09, 1.12, 1.11, 1.05, 1.04, 1.08, 1.13, 1.15, 1.10, 1.01, 1.05, 1.00, 1.05, 1.10, 1.19, 1.17]
val_be_pred_std =        [0.19, 0.21, 0.21, 0.15, 0.17, 0.16, 0.23, 0.28, 0.25, 0.18, 0.18, 0.19, 0.20, 0.17, 0.25, 0.20]

val_design_pred_mean =   [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.04, 0.01, 0.01, 0.02, 0.02, 0.30, 0.42, 0.50]
val_design_pred_std =    [0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.09, 0.00, 0.00, 0.02, 0.00, 0.13, 0.24, 0.18]

val_latent_pred_mean =   [0.07, 0.05, 0.04, 0.05, 0.03, 0.04, 0.05, 0.11, 0.05, 0.03, 0.03, 0.08, 0.14, 0.21, 0.24, 0.58]
val_latent_pred_std =    [0.04, 0.02, 0.02, 0.03, 0.01, 0.03, 0.02, 0.17, 0.04, 0.01, 0.01, 0.07, 0.08, 0.15, 0.12, 0.13]

val_x_pred_mean =        [0.03, 0.03, 0.02, 0.04, 0.03, 0.03, 0.03, 0.05, 0.06, 0.02, 0.03, 0.04, 0.07, 0.36, 0.41, 0.41]
val_x_pred_std =         [0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.06, 0.10, 0.01, 0.01, 0.03, 0.03, 0.03, 0.09, 0.09]

# Scatter plot true and predicted bandgaps
fig, axs = plt.subplots(1, 4, figsize=(12.0, 3.0))
axs[0].errorbar(latent_dims, train_be_pred_mean, yerr=train_be_pred_std, marker='o', c='blue', label='Train binding energy', markersize=4, capsize=3)
axs[0].errorbar(latent_dims, val_be_pred_mean, yerr=val_be_pred_std, marker='o', c='orange', label='Val. binding energy', markersize=4, capsize=3)
axs[0].set_xlabel(r'Latent Dimension')
axs[0].set_xticks(latent_dims)
axs[0].set_xticklabels(latent_dims)
axs[0].set_ylabel(r'MAE Binding Energy (kJ/mol)')
axs[0].legend(frameon=False, fontsize=8)

axs[1].errorbar(latent_dims, train_design_pred_mean, yerr=train_design_pred_std, marker='o', c='blue', label='Train design', markersize=4, capsize=3)
axs[1].errorbar(latent_dims, val_design_pred_mean, yerr=val_design_pred_std, marker='o', c='orange', label='Val design', markersize=4, capsize=3)
axs[1].set_xlabel(r'Latent Dimension')
axs[1].set_xticks(latent_dims)
axs[1].set_xticklabels(latent_dims)
axs[1].set_ylabel(r'Cross Entropy Loss')
# axs[1].set_ylim(0.55, 1.0)
axs[1].legend(frameon=False, fontsize=8)

axs[2].errorbar(latent_dims, train_latent_pred_mean, yerr=train_latent_pred_std, marker='o', c='blue', label='Train latent', markersize=4, capsize=3)
axs[2].errorbar(latent_dims, val_latent_pred_mean, yerr=val_latent_pred_std, marker='o', c='orange', label='Val latent', markersize=4, capsize=3)
axs[2].set_xlabel(r'Latent Dimension')
axs[2].set_xticks(latent_dims)
axs[2].set_xticklabels(latent_dims)
axs[2].set_ylabel(r'MAE Latent Recon.')
# axs[2].set_ylim(0.55, 1.0)
axs[2].legend(frameon=False, fontsize=8)

axs[3].errorbar(latent_dims, train_x_pred_mean, yerr=train_x_pred_std, marker='o', c='blue', label='Train design recon.', markersize=4, capsize=3)
axs[3].errorbar(latent_dims, val_x_pred_mean, yerr=val_x_pred_std, marker='o', c='orange', label='Val design recon.', markersize=4, capsize=3)
axs[3].set_xlabel(r'Latent Dimension')
axs[3].set_xticks(latent_dims)
axs[3].set_xticklabels(latent_dims)
axs[3].set_ylabel(r'MAE Solvent Descriptor Recon.')
# axs[3].set_ylim(0.55, 1.0)
axs[3].legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig('ae2_mhp_binding_energy_latent_size.pdf', bbox_inches='tight', dpi=300)