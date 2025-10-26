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

data = pd.read_csv('data.csv', header=None).to_numpy()

fig, ax = plt.subplots(figsize=(3.0, 3.0))
pcc_plot = ax.matshow(data, cmap='Oranges', vmin=0, vmax=1)
labels = [r'$l_0$', r'$l_1$', r'$l_2$', r'$l_3$', r'$l_4$', r'$l_5$', r'$l_6$', r'$l_7$', r'$l_8$', r'$l_9$', 'bg']
ax.set_yticks(ticks=np.arange(len(labels)), labels=labels, rotation=0)
ax.set_xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90)
cbar = plt.colorbar(pcc_plot, ax=ax, shrink=0.72, pad=0.02)
# cbar.ax.tick_params(labelsize=8)

cbar.set_label(r'$\tilde{\rho}$', rotation=270, labelpad=10)
plt.tight_layout()
plt.savefig('hd1_mhp_latents_and_target_corr_matrix_adj.pdf', format='pdf', dpi=300, bbox_inches='tight')