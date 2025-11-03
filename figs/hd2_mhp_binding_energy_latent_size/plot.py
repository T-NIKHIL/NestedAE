import matplotlib.pyplot as plt

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
# NestedHD
#

latent_dims =            [16,   15,   14,   13,   12,   11,   10,   9,    8,    7,    6,    5,    4,     3,    2]

train_be_pred_mean =     [0.50, 0.61, 0.56, 0.57, 0.65, 0.58, 0.63, 0.62, 0.63, 1.02, 0.72, 0.87, 0.95,  0.97, 1.12]
train_be_pred_std =      [0.04, 0.23, 0.06, 0.04, 0.08, 0.05, 0.07, 0.05, 0.05, 1.09, 0.03, 0.11, 0.08,  0.09, 0.22]

train_design_pred_mean = [0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.04, 0.08, 0.02, 0.10, 0.26,  0.33, 0.45]
train_design_pred_std =  [0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.08, 0.20, 0.00, 0.05, 0.14,  0.12, 0.13]

train_latent_pred_mean = [0.03, 0.05, 0.04, 0.04, 0.06, 0.04, 0.03, 0.03, 0.03, 0.06, 0.02, 0.10, 0.23,  0.33, 0.46]
train_latent_pred_std =  [0.01, 0.04, 0.01, 0.01, 0.03, 0.02, 0.03, 0.01, 0.01, 0.08, 0.00, 0.08, 0.11,  0.09, 0.12]

val_be_pred_mean  =      [1.10, 1.13, 1.09, 1.12, 1.06, 1.06, 1.10, 1.06, 1.11, 1.36, 0.99, 1.11, 1.12,  1.16, 1.17]
val_be_pred_std =        [0.24, 0.26, 0.18, 0.27, 0.20, 0.23, 0.25, 0.24, 0.19, 0.95, 0.20, 0.23, 0.26,  0.23, 0.30]

val_design_pred_mean =   [0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.02, 0.01, 0.05, 0.10, 0.02, 0.11, 0.33,  0.38, 0.51]
val_design_pred_std =    [0.00, 0.01, 0.00, 0.01, 0.02, 0.03, 0.01, 0.00, 0.09, 0.25, 0.01, 0.06, 0.16,  0.16, 0.13]

val_latent_pred_mean =   [0.03, 0.07, 0.05, 0.04, 0.07, 0.05, 0.04, 0.04, 0.04, 0.06, 0.03, 0.10, 0.26,  0.34, 0.48]
val_latent_pred_std =    [0.01, 0.05, 0.02, 0.01, 0.04, 0.03, 0.04, 0.02, 0.01, 0.10, 0.00, 0.08, 0.13,  0.07, 0.14]

# Scatter plot true and predicted bandgaps
fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.0))
axs[0].errorbar(latent_dims, train_be_pred_mean, yerr=train_be_pred_std, marker='o', c='blue', label='Train binding energy', markersize=4, capsize=3)
axs[0].errorbar(latent_dims, val_be_pred_mean, yerr=val_be_pred_std, marker='o', c='orange', label='Val. binding energy', markersize=4, capsize=3)
axs[0].set_xlabel(r'Latent Dimension')
axs[0].set_xticks(latent_dims)
axs[0].set_xticklabels(latent_dims)
axs[0].set_ylabel('MHP-Solvent \n Binding Energy MAE (kJ/mol)')
axs[0].legend(frameon=False, fontsize=8)

axs[1].errorbar(latent_dims, train_design_pred_mean, yerr=train_design_pred_std, marker='o', c='blue', label='Train design', markersize=4, capsize=3)
axs[1].errorbar(latent_dims, val_design_pred_mean, yerr=val_design_pred_std, marker='o', c='orange', label='Val design', markersize=4, capsize=3)
axs[1].set_xlabel(r'Latent Dimension')
axs[1].set_xticks(latent_dims)
axs[1].set_xticklabels(latent_dims)
axs[1].set_ylabel('Design Cross Entropy Loss')
# axs[1].set_ylim(0.55, 1.0)
axs[1].legend(frameon=False, fontsize=8)

axs[2].errorbar(latent_dims, train_latent_pred_mean, yerr=train_latent_pred_std, marker='o', c='blue', label='Train latent', markersize=4, capsize=3)
axs[2].errorbar(latent_dims, val_latent_pred_mean, yerr=val_latent_pred_std, marker='o', c='orange', label='Val latent', markersize=4, capsize=3)
axs[2].set_xlabel(r'Latent Dimension')
axs[2].set_xticks(latent_dims)
axs[2].set_xticklabels(latent_dims)
axs[2].set_ylabel('Latent Recon. MAE')
# axs[2].set_ylim(0.55, 1.0)
axs[2].legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig('hd2_mhp_binding_energy_latent_size.pdf', bbox_inches='tight', dpi=300)