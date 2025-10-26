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
data_no_soft = pd.read_csv('UMAP_coods_no_soft.csv')
data_soft = pd.read_csv('UMAP_coods_soft.csv')

dim1_no_soft = data_no_soft['UMAP1'].values
dim2_no_soft = data_no_soft['UMAP2'].values
pred_bgs_no_soft = data_no_soft['Predicted Bandgaps']

dim1_soft = data_soft['UMAP1'].values
dim2_soft = data_soft['UMAP2'].values
pred_bgs_soft = data_soft['Predicted Bandgaps']

# Scatter plot true and predicted bandgaps
fig, axs = plt.subplots(1, 2, figsize=(6.0, 3.0))
sc1 = axs[0].scatter(dim1_no_soft, dim2_no_soft, c=pred_bgs_no_soft, cmap='viridis', s=1, alpha=1.0, label='No Soft Constraints')
axs[0].set_xlabel(r'UMAP Dimension 1')
axs[0].set_ylabel(r'UMAP Dimension 2')
axs[0].legend(fontsize=8, frameon=False)
# # Set figure size for axs[0] to be same as axs[1]
# axs[0].set_box_aspect(1)
# cbar1 = plt.colorbar(sc1, ax=axs[0])
# cbar1.set_label(r'Predicted Bandgaps (eV)', rotation=270, labelpad=15)
# cbar1.ax.tick_params(labelsize=8)
sc2 = axs[1].scatter(dim1_soft, dim2_soft, c=pred_bgs_soft, cmap='viridis', s=1, alpha=1.0, label='With Soft Constraints')
axs[1].set_xlabel(r'UMAP Dimension 1')
# axs[1].set_ylabel(r'UMAP Dimension 2')
# axs[1].set_box_aspect(1)
axs[1].legend(fontsize=8, frameon=False)
cbar2 = plt.colorbar(sc2, ax=axs[1], pad=0.01)
cbar2.set_label(r'Predicted Bandgaps (eV)', rotation=270, labelpad=10)
cbar2.ax.tick_params(labelsize=8)
# Add title to each subplot
plt.tight_layout()
# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.1)
plt.savefig('hd1_bandgaps_UMAP_comp.pdf', bbox_inches='tight', dpi=300)