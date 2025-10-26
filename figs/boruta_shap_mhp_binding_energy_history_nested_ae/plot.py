import copy

from BorutaShap import BorutaShap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ks_2samp

# https://github.com/Ekeany/Boruta-Shap -> Change to boruta_shap conda env

if __name__ == "__main__":
    # #################################################################
    # --------------------- START OF USER INPUT ---------------------
    # #################################################################

    random_state = 42

    #################################################################
    # Plotting parameters
    #################################################################
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

    #################################################################
    # Dataset Parameters
    #################################################################

    dataset_loc = '../../datasets/MHP_bandgaps_AND_perov_solvent_BE/nestedae_dataset/perov_solv_BE_for_nestedae.csv'
    descriptors = ['A_IONRAD',
                    'A_MASS',
                    'A_DPM',
                    'X_IONRAD',
                    'X_MASS',
                    'X_EA',
                    'X_IE',
                    'X_En',
                    'X_AN',
                    'SOLV_DENSITY',
                    'SOLV_DIELECTRIC',
                    'SOLV_GDN',
                    'SOLV_DPM',
                    'SOLV_MV',
                    'SOLV_UMBO']

    target = ['Target']

    standardize_descs = True
    defined_qs = [0.2, 0.4, 0.6, 0.8]
    train_split = 0.9
    fold_num = 3

    #################################################################
    # Script Parameters
    #################################################################

    n_trials = 200
    train_or_test = 'test'
    load_data = True
    data_filename = 'data.csv'

    # #################################################################
    # --------------------- END OF USER INPUT ---------------------
    # #################################################################

    if not load_data:
        x_dataframe = pd.read_csv(dataset_loc)[descriptors]
        y_dataframe = pd.read_csv(dataset_loc)[target]

        if standardize_descs:
            desc_means_dict = {}
            desc_std_devs = {}
            for desc in x_dataframe.columns.tolist():
                mean = x_dataframe[desc].mean()
                desc_means_dict[desc] = mean
                std_dev = x_dataframe[desc].std()
                desc_std_devs[desc] = std_dev
                x_dataframe[desc] = (x_dataframe[desc] - mean) / std_dev
            print('(INFO) Descriptors standardized.')
        else:
            print('(INFO) Descriptors not standardized.')

        feature_selector = BorutaShap(importance_measure='shap', classification=False)
        feature_selector.fit(X=x_dataframe,
                            y=y_dataframe.to_numpy(dtype=np.float32).reshape(-1, 1), 
                            n_trials=n_trials,
                            sample=False,
                            train_or_test=train_or_test,
                            random_state=random_state,
                            normalize=True,
                            verbose=True,
                            stratify=None) # importance values normaized accdg to Z score
        feature_selector.plot(which_features='all', y_scale=None) # Convert to log by y_scale='log'

        data = feature_selector.history_x.iloc[1:]
        data['index'] = data.index
        data = pd.melt(data, id_vars='index', var_name='Features')

        decision_mapper = feature_selector.create_mapping_of_features_to_attribute(maps=['Tentative','Rejected','Accepted', 'Shadow'])
        data['Decision'] = data['Features'].map(decision_mapper)
        data.drop(['index'], axis=1, inplace=True)

        options = { 'accepted' : feature_selector.filter_data(data,'Decision', 'Accepted'),
                    'tentative': feature_selector.filter_data(data,'Decision', 'Tentative'),
                    'rejected' : feature_selector.filter_data(data,'Decision', 'Rejected'),
                    'all' : data}

        # Save data to csv file
        data.to_csv('data.csv', index=False)

    else:
        # Load the data from csv file
        data = pd.read_csv('data.csv')

    # feats = data.columns.tolist()
    mean_abs_shap_values = []
    bar_colors_dict = {'Accepted': 'green', 'Tentative': 'orange', 'Rejected': 'red', 'Shadow': 'grey'}
    bar_colors = [] 
    for desc in descriptors:
        # Get all the values for the descriptor from the features column
        desc_values = data[data['Methods'] == desc]
        mean_abs_shap_values.append(np.mean(np.abs(desc_values['value'].to_numpy())))
        bar_colors.append(bar_colors_dict[desc_values['Decision'].iloc[0]])

    # # Sort the mean_abs_shap_values and feats_for_plot accordingly in ascending order
    # sorted_indices = np.argsort(mean_abs_shap_values)
    # mean_abs_shap_values_sorted = [mean_abs_shap_values[i] for i in sorted_indices]
    # feats_for_plot = [descriptors[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.barh(descriptors, mean_abs_shap_values, color=bar_colors)
    ax.set_xlabel(r'$ \textrm{Mean} ( | \textrm{SHAP value} | )$')
    ax.set_ylabel(r'Feature')
    plt.tight_layout()
    plt.savefig('borutaSHAP_SHAP_values.pdf', bbox_inches='tight', dpi=300)

