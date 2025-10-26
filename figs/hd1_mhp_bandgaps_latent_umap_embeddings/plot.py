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
data = pd.read_csv('data.csv')

dim1 = data['UMAP1'].values
dim2 = data['UMAP2'].values
pred_bgs = data['Predicted Bandgaps']

# Scatter plot true and predicted bandgaps
fig, ax = plt.subplots(figsize=(4.0, 3.0))
sc = ax.scatter(dim1, dim2, alpha=1, s=1, c=pred_bgs, cmap='viridis')
ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Predicted Bandgap (eV)')
plt.tight_layout()
plt.savefig('hd1_mhp_latent_umap_embeddings.pdf', bbox_inches='tight', dpi=300)