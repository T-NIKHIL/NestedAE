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
latent_dims =                    [15,   14,   13,   12,   11,   10,   9,    8,    7,    6,    5,    4]
# This is old data need to rerun these expriments if data is needed.
# train_bg_pred_loss_no_soft =     [0.09, 0.09, 0.07, 0.08, 0.08, 0.09, 0.10, 0.10, 0.09, 0.11, 0.1]
# train_design_pred_loss_no_soft = [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.75, 0.75, 0.75, 0.84, 0.94]
# val_bg_pred_loss_no_soft =       [0.16, 0.17, 0.19, 0.19, 0.18, 0.20, 0.17, 0.18, 0.18, 0.18, 0.18]
# val_design_pred_loss_no_soft =   [0.57, 0.57, 0.57, 0.56, 0.56, 0.57, 0.59, 0.58, 0.58, 0.70, 0.77]

train_bg_pred_loss_soft_mean =   [0.07, 0.08, 0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.10, 0.11, 0.12, 0.13]
train_bg_pred_loss_soft_std  =   [0.01, 0.00, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01]

train_design_pred_loss_soft_mean=[0.61, 0.62, 0.63, 0.62, 0.64, 0.63, 0.65, 0.66, 0.67, 0.71, 0.75, 0.78]
train_design_pred_loss_soft_std =[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01]

val_bg_pred_loss_soft_mean =     [0.16, 0.17, 0.17, 0.16, 0.17, 0.15, 0.16, 0.15, 0.15, 0.16, 0.16, 0.16]
val_bg_pred_loss_soft_std  =     [0.03, 0.03, 0.04, 0.03, 0.04, 0.03, 0.03, 0.03, 0.03, 0.05, 0.04, 0.03]

val_design_pred_loss_soft_mean = [0.69, 0.69, 0.73, 0.69, 0.73, 0.69, 0.72, 0.73, 0.73, 0.78, 0.85, 0.86]
val_design_pred_loss_soft_std  = [0.09, 0.09, 0.09, 0.06, 0.08, 0.08, 0.08, 0.07, 0.07, 0.08, 0.08, 0.09]


# Scatter plot true and predicted bandgaps
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
axs[0].errorbar(latent_dims, train_bg_pred_loss_soft_mean, yerr=train_bg_pred_loss_soft_std, marker='o', c='blue', label='Train bandgaps', markersize=4, capsize=3)
axs[0].errorbar(latent_dims, val_bg_pred_loss_soft_mean, yerr=val_bg_pred_loss_soft_std, marker='o', c='orange', label='Val bandgaps', markersize=4, capsize=3)

axs[1].errorbar(latent_dims, train_design_pred_loss_soft_mean, yerr=train_design_pred_loss_soft_std, marker='o', c='blue', label='Train design', markersize=4, capsize=3)
axs[1].errorbar(latent_dims, val_design_pred_loss_soft_mean, yerr=val_design_pred_loss_soft_std, marker='o', c='orange', label='Val design', markersize=4, capsize=3)

axs[0].set_xlabel(r'Latent Dimension')
axs[0].set_xticks(latent_dims)
axs[0].set_xticklabels(latent_dims)
axs[1].set_xlabel(r'Latent Dimension')
axs[1].set_xticks(latent_dims)
axs[1].set_xticklabels(latent_dims)
# Set left side y label to be MAE bandgap and other side to be cross entropy loss
axs[0].set_ylabel(r'MAE Bandgap (eV)')
# ax2 = axs[0].twinx()
axs[1].set_ylabel(r'Cross Entropy Loss')
axs[1].set_ylim(0.55, 1.0)
axs[0].legend(frameon=False, fontsize=8)
axs[1].legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig('hd1_mhp_bandgaps_latent_size_comparison.pdf', bbox_inches='tight', dpi=300)