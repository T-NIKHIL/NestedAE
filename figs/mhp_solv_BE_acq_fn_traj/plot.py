import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch import from_numpy

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

which_trial = 2
random_state = 42
which_latent_hist_file = 'selected_latent_hist_no_feat_sel_min.npy'
save_video_filename = f'mhp_solv_BE_acq_fn_traj_trial_{which_trial + 1}_no_feat_sel_min.mp4'
maximize = False

# Load the data from csv file
all_data_umap = pd.read_csv('all_data_umap.csv')
dim1 = all_data_umap['UMAP1'].values
dim2 = all_data_umap['UMAP2'].values
true_be = all_data_umap['True Binding Energy'].values

# Load the x_torch_all and selected_latent_hist numpy arrays
x_torch_all = from_numpy(np.load('x_torch_all.npy'))
selected_latent_hist = np.load(which_latent_hist_file, allow_pickle=True)

sel_data_umap_list = []
latents = selected_latent_hist[which_trial, :]
for latent in latents:
    match_idx = (x_torch_all == from_numpy(latent)).all(dim=1).nonzero(as_tuple=False).item()
    sel_data_umap_list.append(all_data_umap[['UMAP1', 'UMAP2']].values[match_idx])

sel_data_umap = pd.DataFrame(np.array(sel_data_umap_list), columns=['UMAP1', 'UMAP2'])

for i in range(len(sel_data_umap)):
    if maximize:
        umap_dim1_opt_be = all_data_umap['UMAP1'].values[np.argmax(true_be)]
        umap_dim2_opt_be = all_data_umap['UMAP2'].values[np.argmax(true_be)]
    else:
        umap_dim1_opt_be = all_data_umap['UMAP1'].values[np.argmin(true_be)]
        umap_dim2_opt_be = all_data_umap['UMAP2'].values[np.argmin(true_be)]
    fig, ax = plt.subplots(1,1, figsize=(4, 3))
    sc = ax.scatter(dim1, dim2, c=true_be, cmap='viridis', s=5, alpha=0.5, edgecolors='none')
    cbar = plt.colorbar(sc)
    ax.scatter(umap_dim1_opt_be, umap_dim2_opt_be, color='black', marker='*', s=10, label='Candidate with min. BE')
    cbar.set_label('True Binding Energy', rotation=270, labelpad=15)
    ax.scatter(sel_data_umap['UMAP1'].values[i], sel_data_umap['UMAP2'].values[i], color='red', s=10, label='Candidate selected by EI')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    # Remove box frame
    plt.legend(fontsize=6, frameon=False)
    plt.title(f'Trial {which_trial + 1}, Step {i+1}', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{i}.png', bbox_inches='tight', dpi=300)
    plt.close()

# os.system('for i in {0..99}; do files="$files ${i}.pdf"; done; pdfunite $files combined.pdf')
# os.system('for i in {0..99}; do rm ${i}.pdf; done;')
# Convert to video
os.system(f'ffmpeg -framerate 5 -i %d.png -c:v libx264 -r 30 -pix_fmt yuv420p {save_video_filename}')
os.system('for i in {0..99}; do rm ${i}.png; done;')

