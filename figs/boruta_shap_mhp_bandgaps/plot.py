import copy

from BorutaShap import BorutaShap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

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

    dataset_loc = '../../datasets/MHP_bandgaps_AND_perov_solvent_BE/perov_bandgaps_PBE.csv'
    descriptors = ['A_IONRAD',
                   'A_MASS',
                   'A_DPM',
                   'B_IONRAD',
                   'B_MASS',
                   'B_EA',
                   'B_IE',
                   'B_En',
                   'B_AN', 
                   'X_IONRAD',
                   'X_MASS',
                   'X_EA',
                   'X_IE',
                   'X_En',
                   'X_AN']
    target = ['Gap']
    standardize_descs = True

    #################################################################
    # Script Parameters
    #################################################################

    n_trials = 200
    train_or_test = 'test'
    load_data = True
    data_filename = 'data.csv'
    rename_columns = None

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

        # reg = GradientBoostingRegressor(loss='absolute_error',
        #                                 learning_rate=0.05,
        #                                 n_estimators=1000,
        #                                 max_depth=4,
        #                                 subsample=0.6,
        #                                 ccp_alpha=0.01,
        #                                 random_state=random_state)
        reg = GradientBoostingRegressor(loss='absolute_error',
                                        random_state=random_state)
        feature_selector = BorutaShap(model=reg, importance_measure='shap', classification=False)
        feature_selector.fit(X=x_dataframe,
                            y=y_dataframe.to_numpy(dtype=np.float32).reshape(-1, 1), 
                            n_trials=n_trials,
                            sample=False,
                            train_or_test=train_or_test,
                            random_state=random_state,
                            normalize=True,
                            verbose=True,
                            stratify=None) # importance values normalized accdg to Z score
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

    if rename_columns is None:
        rename_columns = descriptors

    fig, ax = plt.subplots(figsize=(4.3, 3))
    bar_colors_dict = {'Accepted':'green', 'Rejected':'red', 'Tentative':'orange'}
    k = 1
    for (rename_col, desc) in zip(rename_columns, descriptors):
        z_scores = data[data['Features'] == desc]
        z_scores['value'] = z_scores['value'].fillna(0.0)
        z_scores_arr = z_scores['value'].to_numpy()
        color = bar_colors_dict[z_scores['Decision'].iloc[0]]
        bplot = ax.boxplot(z_scores_arr, vert=False, positions=[k], widths=0.7, 
                            medianprops=dict(color='black'),
                            flierprops=dict(markerfacecolor='black', marker='.', markersize=2, markeredgecolor='k'),
                            capprops=dict(color='k', linewidth=1),
                            labels=[rename_col])
        plt.setp(ax.get_yticklabels()[-1], color=color)
        k += 1

    ax.set_xlabel(r'$ \textrm{Z-score}$')
    ax.set_ylabel(r'Feature')
    ax.set_xticks(np.arange(-0.5, 5.5, 1))
    plt.tight_layout()
    plt.savefig('borutaSHAP_values.pdf', bbox_inches='tight', dpi=300)

