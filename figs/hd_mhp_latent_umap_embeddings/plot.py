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
data = pd.read_csv('data_with_pred_values.csv')

dim1 = data['UMAP1'].values
dim2 = data['UMAP2'].values
pred_bgs = data['Predicted Bandgaps'].values
pred_be = data['Predicted Binding Energy'].values

# Scatter plot true and predicted bandgaps
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
sc = ax[0].scatter(dim1, dim2, alpha=1.0, s=1, c=pred_bgs, cmap='viridis')
ax[0].set_xlabel('UMAP Dimension 1')
ax[0].set_ylabel('UMAP Dimension 2')
ax[0].set_xticks(np.arange(-5, 5, 2.5))
# ax[0].set_yticks(np.arange(0, 17.5, 2.5))
# ax[0].set_aspect('equal', 'box')
cbar = plt.colorbar(sc, ax=ax[0])
cbar.set_label('Predicted Bandgap (eV)')
sc1 = ax[1].scatter(dim1, dim2, alpha=1.0, s=1, c=pred_be, cmap='viridis')
ax[1].set_xlabel('UMAP Dimension 1')
ax[1].set_ylabel('UMAP Dimension 2')
ax[1].set_xticks(np.arange(-5, 5, 2.5))
# ax[1].set_yticks(np.arange(-5, 17.5, 5.0))
# ax[1].set_aspect('equal', 'box')
cbar = plt.colorbar(sc1, ax=ax[1])
cbar.set_label('Predicted Binding Energy (eV)')
plt.tight_layout()
plt.savefig('hd_mhp_latent_umap_embeddings.pdf', bbox_inches='tight', dpi=300)