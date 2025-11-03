import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Plotting parameters
plt.rcParams.update({
"text.usetex":True,
"font.family":"sans-serif",
"font.serif":["Computer Modern Roman"]})
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.minor.width'] = 1

plot_conditioned_ae = True

latent_dims                      = [26,   25,   24,   23,   22,   21,   20,   19,   18,   17,    16,    15,    14,   13,   12,   11,   10,   9,    8]

train_y_pred_loss_mean           = [0.19, 0.14, 0.15, 0.17, 0.16, 0.15, 0.18, 0.14, 0.15, 0.16,  0.18,  0.16,  0.16, 0.18, 0.17, 0.16, 0.23, 0.16, 0.24]
train_y_pred_loss_std            = [0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.04, 0.01, 0.02, 0.03,  0.05,  0.01,  0.02, 0.03, 0.04, 0.01, 0.06, 0.03, 0.04]

train_design_pred_loss_mean      = [1.63, 1.62, 1.65, 1.69, 1.65, 1.66, 1.70, 1.68, 1.71, 1.70,  1.80,  1.81,  1.78, 1.80, 1.84, 1.81, 1.94, 1.88, 2.01]
train_design_pred_loss_std       = [0.02, 0.03, 0.01, 0.05, 0.02, 0.02, 0.07, 0.02, 0.02, 0.02,  0.03,  0.02,  0.07, 0.04, 0.04, 0.02, 0.04, 0.03, 0.02]

train_x_pred_loss_mean           = [0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.08,  0.08,  0.09,  0.09, 0.09, 0.09, 0.09, 0.10, 0.11, 0.12]
train_x_pred_loss_std            = [0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  0.01,  0.01,  0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.00]

val_y_pred_loss_mean             = [0.17, 0.15, 0.17, 0.17, 0.18, 0.16, 0.18, 0.16, 0.15, 0.18,  0.16,  0.17,  0.17, 0.20, 0.16, 0.17, 0.24, 0.17, 0.23]
val_y_pred_loss_std              = [0.02, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.02, 0.02,  0.03,  0.02,  0.04, 0.06, 0.02, 0.02, 0.04, 0.04, 0.05]

val_design_pred_loss_mean        = [1.88, 1.89, 1.91, 1.92, 1.88, 1.87, 1.91, 1.91, 1.91, 1.92,  1.96,  1.98,  1.93, 1.95, 1.99, 1.98, 2.09, 2.01, 2.12]
val_design_pred_loss_std         = [0.06, 0.07, 0.04, 0.08, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,  0.03,  0.04,  0.06, 0.03, 0.03, 0.04, 0.07, 0.05, 0.03]

val_x_pred_loss_mean             = [0.07, 0.07, 0.08, 0.07, 0.07, 0.07, 0.08, 0.08, 0.09, 0.08,  0.08,  0.09,  0.09, 0.10, 0.09, 0.09, 0.10, 0.11, 0.13]
val_x_pred_loss_std              = [0.00, 0.01, 0.00, 0.01, 0.00, 0.01, 0.00, 0.01, 0.01, 0.00,  0.01,  0.01,  0.00, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]


# Scatter plot true and predicted bandgaps
fig, axs = plt.subplots(1, 3, figsize=(11, 3))
axs[0].errorbar(latent_dims, train_y_pred_loss_mean, yerr=train_y_pred_loss_std, marker='o', c='blue', label='Train y Prediction', markersize=3, capsize=2)
axs[0].errorbar(latent_dims, val_y_pred_loss_mean, yerr=val_y_pred_loss_std, marker='o', c='orange', label='Val y Prediction', markersize=3, capsize=2)

axs[1].errorbar(latent_dims, train_design_pred_loss_mean, yerr=train_design_pred_loss_std, marker='o', c='blue', label='Train Design Prediction', markersize=3, capsize=2)
axs[1].errorbar(latent_dims, val_design_pred_loss_mean, yerr=val_design_pred_loss_std, marker='o', c='orange', label='Val Design Prediction', markersize=3, capsize=2)

axs[2].errorbar(latent_dims, train_x_pred_loss_mean, yerr=train_x_pred_loss_std, marker='o', c='blue', label='Train X Prediction', markersize=3, capsize=2)
axs[2].errorbar(latent_dims, val_x_pred_loss_mean, yerr=val_x_pred_loss_std, marker='o', c='orange', label='Val X Prediction', markersize=3, capsize=2)

axs[0].set_xlabel(r'Latent Dimension')
axs[0].set_xticks(latent_dims)
axs[0].set_xticklabels(latent_dims)

axs[1].set_xlabel(r'Latent Dimension')
axs[1].set_xticks(latent_dims)
axs[1].set_xticklabels(latent_dims)

axs[2].set_xlabel(r'Latent Dimension')
axs[2].set_xticks(latent_dims)
axs[2].set_xticklabels(latent_dims)

axs[0].set_ylabel(r'STHE MAE ( units of \% )')
axs[1].set_ylabel(r'Design Cross Entropy Loss')
axs[2].set_ylabel(r'Descriptor Recon. MAE')

axs[0].legend(frameon=False, fontsize=8)
axs[1].legend(frameon=False, fontsize=8)
axs[2].legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig('singleae_mhp_STHE_latent_size_comp.pdf', bbox_inches='tight', dpi=300)
