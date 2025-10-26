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

feats = data.columns.tolist()
feats_for_plot = [r'$\textrm{A}_{\textrm{ionrad}}$', r'$\textrm{A}_{\textrm{mass}}$', r'$\textrm{A}_{\textrm{dipole}}$', \
                  r'$\textrm{B}_{\textrm{ionrad}}$', r'$\textrm{B}_{\textrm{mass}}$', r'$\textrm{B}_{\textrm{EA}}$', r'$\textrm{B}_{\textrm{IE}}$', r'$\textrm{B}_{\textrm{En}}$', r'$\textrm{B}_{\textrm{AN}}$', \
                  r'$\textrm{X}_{\textrm{ionrad}}$', r'$\textrm{X}_{\textrm{mass}}$', r'$\textrm{X}_{\textrm{EA}}$', r'$\textrm{X}_{\textrm{IE}}$', r'$\textrm{X}_{\textrm{En}}$', r'$\textrm{X}_{\textrm{AN}}$']
mean_abs_shapley_values = []

for feat in feats:
    mean_abs_shapley_values.append(np.mean(np.abs(data[feat])))

mean_abs_shapley_values, feats_for_plot = zip(*sorted(zip(mean_abs_shapley_values, feats_for_plot), reverse=False))

fig, ax = plt.subplots(figsize=(4, 3))
ax.barh(feats_for_plot, mean_abs_shapley_values, color='skyblue')
ax.set_xlabel(r'$ \textrm{Mean} ( | \textrm{shapley value} | )$')
ax.set_ylabel(r'Feature')
plt.tight_layout()
plt.savefig('ae1_mhp_bandgaps_exact_shapley_values.pdf', bbox_inches='tight', dpi=300)

