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
model_name =     ['XgBoost', 'RandomForest', 'NuSVR', 'Lasso', 'Ridge', 'NestedAE', 'NestedHD']
mean_mae_test =  [1.014,      1.055,          0.972,   1.197,   1.196,   1.050,      0.990]
std_mae_test =   [0.174,      0.161,          0.139,   0.147,   0.152,   0.180,      0.200]
mean_mae_train = [0.796,      0.761,          0.833,   1.137,   1.135,   0.660,      0.720]
std_mae_train =  [0.027,      0.021,          0.017,   0.023,   0.023,   0.040,      0.030]

# Scatter plot true and predicted bandgaps
fig, ax = plt.subplots(figsize=(4.0, 3.0))
# Create plots to show mean and standard deviation of MAE for test and train datasets
ax.errorbar(model_name, y=mean_mae_train, yerr=std_mae_train, label='Train', fmt='o', color='blue', capsize=5)
ax.errorbar(model_name, y=mean_mae_test, yerr=std_mae_test, label='Val', fmt='o', color='orange', capsize=5)
ax.set_xticks(np.arange(len(model_name)))
ax.set_xticklabels(model_name, rotation=45, ha='right')
ax.set_ylabel('Mean Absolute Error (eV)')
plt.tight_layout()
plt.legend(frameon=False, fontsize=8)
plt.savefig('NAE_v_NHD_v_other_ML_mhp_solv_BE.pdf', bbox_inches='tight', dpi=300)