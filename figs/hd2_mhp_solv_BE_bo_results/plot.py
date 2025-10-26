import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

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

which_batches_to_reach_optimal_file = 'batches_to_reach_optimal_nestedhd_min.csv'
save_fig_filename = 'space_explored_nestedhd_min.pdf'

# Load the data from csv file
data = pd.read_csv(which_batches_to_reach_optimal_file)

fig, ax = plt.subplots(1,1, figsize=(4, 3))
ax = sns.boxplot(x='method', y='value', data=data, 
                showfliers=True, width=0.5,
                medianprops=dict(color='black', linewidth=0.5))
# This is to display the dots 
ax = sns.stripplot(x='method', y='value', data=data, 
                   hue='method', legend=False,
                   alpha=.4, linewidth=1, jitter=0)

mean_val = data['value'].mean()
median_val = data['value'].median()
ax.text(-0.4, mean_val + 0.6, f"Mean: {mean_val:.1f}", ha='center', fontsize=8, color='black', fontweight='bold')
# ax.text(-0.4, median_val + 0.1, f"Median: {median_val:.1f}", ha='center', fontsize=8, color='black', fontweight='bold')
# Show mean as a horizontal dotted red line
ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label='Mean')
# # Show median as a horizontal dotted black line
# ax.axhline(median_val, color='black', linestyle='--', linewidth=2, label='Median')
# Set y axis values
ax.set_yticks(np.arange(0, 40, 2))
ax.set_xticks([])
ax.set_xticklabels([])

plt.gca().set_xlabel('')
plt.gca().set_ylabel(r'\% space explored')
plt.tight_layout()
plt.savefig(save_fig_filename, bbox_inches='tight', dpi=300)