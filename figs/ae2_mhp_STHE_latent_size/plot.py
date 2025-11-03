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

plot_conditioned_ae = True

latent_dims                      = [14,   13,   12,   11,   10,   9,    8,    7,    6]

train_y_pred_loss_mean           = [0.11, 0.12, 0.12, 0.17, 0.11, 0.16, 0.16, 0.17, 0.18]
train_y_pred_loss_std            = [0.01, 0.02, 0.01, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01]

train_design_pred_loss_mean      = [0.56, 0.58, 0.59, 0.64, 0.62, 0.63, 0.65, 0.66, 0.81]
train_design_pred_loss_std       = [0.01, 0.02, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.03]

train_latent_pred_loss_mean      = [0.04, 0.05, 0.05, 0.05, 0.06, 0.05, 0.06, 0.20, 0.40]
train_latent_pred_loss_std       = [0.00, 0.01, 0.01, 0.00, 0.01, 0.01, 0.00, 0.01, 0.00]

train_x_pred_loss_mean           = [0.07, 0.08, 0.09, 0.16, 0.10, 0.12, 0.22, 0.23, 0.28]
train_x_pred_loss_std            = [0.00, 0.01, 0.00, 0.03, 0.01, 0.00, 0.00, 0.00, 0.05]

val_y_pred_loss_mean             = [0.12, 0.12, 0.13, 0.16, 0.12, 0.16, 0.15, 0.17, 0.18]
val_y_pred_loss_std              = [0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.02]

val_design_pred_loss_mean        = [0.76, 0.73, 0.75, 0.78, 0.79, 0.77, 0.79, 0.77, 0.89]
val_design_pred_loss_std         = [0.04, 0.05, 0.05, 0.03, 0.07, 0.05, 0.04, 0.03, 0.06]

val_latent_pred_loss_mean        = [0.04, 0.05, 0.05, 0.05, 0.06, 0.05, 0.06, 0.20, 0.40]
val_latent_pred_loss_std         = [0.01, 0.01, 0.01, 0.00, 0.01, 0.01, 0.01, 0.01, 0.01]

val_x_pred_loss_mean             = [0.07, 0.08, 0.10, 0.16, 0.10, 0.12, 0.22, 0.23, 0.28]
val_x_pred_loss_std              = [0.01, 0.01, 0.01, 0.03, 0.01, 0.01, 0.01, 0.01, 0.05]


# Scatter plot true and predicted bandgaps
fig, axs = plt.subplots(1, 3, figsize=(11, 3))
axs[0].errorbar(latent_dims, train_y_pred_loss_mean, yerr=train_y_pred_loss_std, marker='o', c='blue', label='Train y Prediction', markersize=4, capsize=3)
axs[0].errorbar(latent_dims, val_y_pred_loss_mean, yerr=val_y_pred_loss_std, marker='o', c='orange', label='Val y Prediction', markersize=4, capsize=3)

axs[1].errorbar(latent_dims, train_design_pred_loss_mean, yerr=train_design_pred_loss_std, marker='o', c='blue', label='Train Design Prediction', markersize=4, capsize=3)
axs[1].errorbar(latent_dims, val_design_pred_loss_mean, yerr=val_design_pred_loss_std, marker='o', c='orange', label='Val Design Prediction', markersize=4, capsize=3)

# axs[1, 0].errorbar(latent_dims, train_latent_pred_loss_mean, yerr=train_latent_pred_loss_std, marker='o', c='blue', label='Train Latent Prediction', markersize=4, capsize=3)
# axs[1, 0].errorbar(latent_dims, val_latent_pred_loss_mean, yerr=val_latent_pred_loss_std, marker='o', c='orange', label='Val Latent Prediction', markersize=4, capsize=3)

axs[2].errorbar(latent_dims, train_x_pred_loss_mean, yerr=train_x_pred_loss_std, marker='o', c='blue', label='Train X Prediction', markersize=4, capsize=3)
axs[2].errorbar(latent_dims, val_x_pred_loss_mean, yerr=val_x_pred_loss_std, marker='o', c='orange', label='Val X Prediction', markersize=4, capsize=3)

axs[0].set_xlabel(r'Latent Dimension')
axs[0].set_xticks(latent_dims)
axs[0].set_xticklabels(latent_dims)

axs[1].set_xlabel(r'Latent Dimension')
axs[1].set_xticks(latent_dims)
axs[1].set_xticklabels(latent_dims)

axs[2].set_xlabel(r'Latent Dimension')
axs[2].set_xticks(latent_dims)
axs[2].set_xticklabels(latent_dims)

# axs[1, 1].set_xlabel(r'Latent Dimension')
# axs[1, 1].set_xticks(latent_dims)
# axs[1, 1].set_xticklabels(latent_dims)

axs[0].set_ylabel(r'STHE MAE ( in units of \% )')
axs[1].set_ylabel(r'Design Cross Entropy Loss')
# axs[1, 0].set_ylabel(r'Latent Recon. MAE')
axs[2].set_ylabel(r'Descriptor Recon. MAE')

axs[0].legend(frameon=False, fontsize=8)
axs[1].legend(frameon=False, fontsize=8)
axs[2].legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig('ae2_mhp_STHE_latent_size_comp.pdf', bbox_inches='tight', dpi=300)
