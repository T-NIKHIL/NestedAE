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

adj_pcc_no_soft = pd.read_csv('PCC_no_soft.csv', header=None).to_numpy()
adj_pcc_soft = pd.read_csv('PCC_soft.csv', header=None).to_numpy()

fig, ax = plt.subplots(1, 2, figsize=(6.0, 3.0))
pcc_plot1 = ax[0].matshow(adj_pcc_no_soft, cmap='Oranges', vmin=0, vmax=1)
labels1 = [r'$l_0$', r'$l_1$', r'$l_2$', r'$l_3$', r'$l_4$', r'$l_5$', r'$l_6$', r'$l_7$', r'$l_8$', r'$l_9$', 'bg']
labels2 = [r'$l_0$', r'$l_1$', r'$l_2$', r'$l_3$', r'$l_4$', r'$l_5$', r'$l_6$', r'$l_7$', r'$l_8$', 'bg']
ax[0].set_yticks(ticks=np.arange(len(labels1)), labels=labels1, rotation=0)
ax[0].set_xticks(ticks=np.arange(len(labels1)), labels=labels1, rotation=90)
# cbar.ax.tick_params(labelsize=8)
pcc_plot2 = ax[1].matshow(adj_pcc_soft, cmap='Oranges', vmin=0, vmax=1)
ax[1].set_yticks(ticks=np.arange(len(labels2)), labels=labels2, rotation=0)
ax[1].set_xticks(ticks=np.arange(len(labels2)), labels=labels2, rotation=90)
# cbar2.ax.tick_params(labelsize=8)
# Add title to each subplot
ax[0].set_title('No Soft Constraints', pad=10)
ax[1].set_title('With Soft Constraints', pad=10)    
cbar2 = plt.colorbar(pcc_plot2, ax=ax[1], shrink=1, pad=0.05)
cbar2.set_label(r'$\tilde{\rho}$', rotation=270, labelpad=10)
plt.tight_layout()
plt.savefig('ae1_mhp_latents_and_target_corr_matrix_adj_comp.pdf', format='pdf', dpi=300, bbox_inches='tight')