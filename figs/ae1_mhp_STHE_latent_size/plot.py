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

latent_dims                      = [19,   18,   17,   16,   15,   10,   9,    8,    7,    6,    5,   4]

train_design_pred_loss_mean      = [1.20, 1.18, 1.19, 1.18, 1.18, 1.20, 1.19, 1.20, 1.19, 1.24, 1.30, 1.32]
train_design_pred_loss_std       = [0.03, 0.00, 0.02, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.04, 0.01, 0.01]

train_x_pred_loss_mean           = [0.03, 0.03, 0.04, 0.04, 0.03, 0.05, 0.05, 0.04, 0.05, 0.11, 0.13, 0.15]
train_x_pred_loss_std            = [0.00, 0.00, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.01, 0.00]

val_design_pred_loss_mean        = [1.21, 1.18, 1.19, 1.19, 1.18, 1.21, 1.20, 1.22, 1.20, 1.24, 1.31, 1.33]
val_design_pred_loss_std         = [0.03, 0.03, 0.02, 0.03, 0.03, 0.02, 0.02, 0.05, 0.03, 0.03, 0.03, 0.03]

val_x_pred_loss_mean             = [0.03, 0.03, 0.04, 0.04, 0.03, 0.05, 0.05, 0.04, 0.05, 0.11, 0.13, 0.15]
val_x_pred_loss_std              = [0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01]

# Losses with conditioned Autoencoder

latent_dims_for_cond             = [19,   18,   17,   16,   15,   14,   13,   12,   11,   10,   9,    8,    7,    6]

train_design_pred_cond_loss_mean = [1.18, 1.19, 1.18, 1.18, 1.19, 1.19, 1.19, 1.19, 1.20, 1.20, 1.21, 1.23, 1.23, 1.28]
train_design_pred_cond_loss_std  = [0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.01]

train_x_pred_cond_loss_mean      = [0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.05, 0.06, 0.05, 0.06, 0.06, 0.06, 0.08]
train_x_pred_cond_loss_std       = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.01, 0.01]

train_STHE_pred_cond_loss_mean   = [0.60, 0.73, 0.63, 0.66, 0.71, 0.71, 0.83, 0.78, 0.86, 0.87, 0.89, 0.93, 0.98, 1.10]
train_STHE_pred_cond_loss_std    = [0.01, 0.05, 0.02, 0.04, 0.05, 0.05, 0.04, 0.03, 0.12, 0.07, 0.02, 0.03, 0.05, 0.08]

val_design_pred_cond_loss_mean   = [1.19, 1.20, 1.19, 1.18, 1.19, 1.20, 1.20, 1.19, 1.21, 1.21, 1.22, 1.23, 1.24, 1.29]
val_design_pred_cond_loss_std    = [0.02, 0.03, 0.02, 0.02, 0.03, 0.03, 0.02, 0.02, 0.06, 0.02, 0.02, 0.02, 0.04, 0.03]

val_x_pred_cond_loss_mean        = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.06, 0.08]
val_x_pred_cond_loss_std         = [0.00, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

val_STHE_pred_cond_loss_mean     = [0.73, 0.87, 0.79, 0.81, 0.84, 0.86, 0.95, 0.95, 0.99, 0.96, 1.02, 1.02, 1.08, 1.21]
val_STHE_pred_cond_loss_std      = [0.06, 0.04, 0.07, 0.07, 0.07, 0.06, 0.07, 0.06, 0.17, 0.10, 0.05, 0.08, 0.07, 0.09]

# Scatter plot true and predicted bandgaps
if not plot_conditioned_ae:
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.0))
    axs[0].errorbar(latent_dims, train_design_pred_loss_mean, yerr=train_design_pred_loss_std, marker='o', c='blue', label='Train Design Prediction', markersize=4, capsize=3, linestyle='None')
    axs[0].errorbar(latent_dims, val_design_pred_loss_mean, yerr=val_design_pred_loss_std, marker='o', c='orange', label='Val Design Prediction', markersize=4, capsize=3, linestyle='None')

    axs[1].errorbar(latent_dims, train_x_pred_loss_mean, yerr=train_x_pred_loss_std, marker='o', c='blue', label='Train X Reconstruction', markersize=4, capsize=3, linestyle='None')
    axs[1].errorbar(latent_dims, val_x_pred_loss_mean, yerr=val_x_pred_loss_std, marker='o', c='orange', label='Val X Reconstruction', markersize=4, capsize=3, linestyle='None')

    axs[0].set_xlabel(r'Latent Dimension')
    axs[0].set_xticks(latent_dims)
    axs[0].set_xticklabels(latent_dims)
    axs[1].set_xlabel(r'Latent Dimension')
    axs[1].set_xticks(latent_dims)
    axs[1].set_xticklabels(latent_dims)
    # Set left side y label to be MAE bandgap and other side to be cross entropy loss
    axs[0].set_ylabel(r'Cross Entropy Loss')
    # ax2 = axs[0].twinx()
    axs[1].set_ylabel(r'MAE Reconstruction')
    # axs[1].set_ylim(0.55, 1.0)
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].legend(frameon=False, fontsize=8)
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -5*d), (1, 5*d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    axs[0].plot([0.55, 0.57], [0, 0], transform=axs[0].transAxes, **kwargs)
    axs[1].plot([0.55, 0.57], [0, 0], transform=axs[1].transAxes, **kwargs)
    plt.tight_layout()
    plt.savefig('ae1_mhp_STHE_latent_size_comp.pdf', bbox_inches='tight', dpi=300)
else:
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.0))
    axs[0].errorbar(latent_dims_for_cond, train_design_pred_cond_loss_mean, yerr=train_design_pred_cond_loss_std, marker='o', c='blue', label='Train Design Prediction', markersize=4, capsize=3)
    axs[0].errorbar(latent_dims_for_cond, val_design_pred_cond_loss_mean, yerr=val_design_pred_cond_loss_std, marker='o', c='orange', label='Val Design Prediction', markersize=4, capsize=3)
    axs[1].errorbar(latent_dims_for_cond, train_x_pred_cond_loss_mean, yerr=train_x_pred_cond_loss_std, marker='o', c='blue', label='Train X Reconstruction', markersize=4, capsize=3)
    axs[1].errorbar(latent_dims_for_cond, val_x_pred_cond_loss_mean, yerr=val_x_pred_cond_loss_std, marker='o', c='orange', label='Val X Reconstruction', markersize=4, capsize=3)
    axs[2].errorbar(latent_dims_for_cond, train_STHE_pred_cond_loss_mean, yerr=train_STHE_pred_cond_loss_std, marker='o', c='blue', label='Train STHE Prediction', markersize=4, capsize=3)
    axs[2].errorbar(latent_dims_for_cond, val_STHE_pred_cond_loss_mean, yerr=val_STHE_pred_cond_loss_std, marker='o', c='orange', label='Val STHE Prediction', markersize=4, capsize=3)

    axs[0].set_xlabel(r'Latent Dimension')
    axs[0].set_xticks(latent_dims_for_cond)
    axs[0].set_xticklabels(latent_dims_for_cond)

    axs[1].set_xlabel(r'Latent Dimension')
    axs[1].set_xticks(latent_dims_for_cond)
    axs[1].set_xticklabels(latent_dims_for_cond)

    axs[2].set_xlabel(r'Latent Dimension')
    axs[2].set_xticks(latent_dims_for_cond)
    axs[2].set_xticklabels(latent_dims_for_cond)

    axs[0].set_ylabel(r'Cross Entropy Loss')
    axs[1].set_ylabel(r'MAE Reconstruction')
    axs[2].set_ylabel(r'MAE STHE Prediction')

    axs[0].legend(frameon=False, fontsize=8)
    axs[1].legend(frameon=False, fontsize=8)
    axs[2].legend(frameon=False, fontsize=8)

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    plt.tight_layout()
    plt.savefig('ae1_cond_mhp_STHE_latent_size_comp.pdf', bbox_inches='tight', dpi=300)
