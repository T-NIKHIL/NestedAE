import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from scipy.stats import ks_2samp

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
    defined_qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    train_split = 0.9

    param_grid_xgb = {
        'n_estimators': [250, 500, 1000, 2000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.6, 0.8, 1.0],
        # 'colsample_bytree': [0.6, 0.8, 1.0],
        'alpha': [1.0],
        # 'lambda':[0, 0.01, 0.1]
    }
    param_grid_RF = {
        'n_estimators': [100, 200, 500, 1000],
        'max_depth': [2, 3, 4, 5, 10],
        'ccp_alpha': [0.0, 0.01, 0.1]
    }
    param_grid_nusvr = {
        'nu': [0.25, 0.5, 0.75],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'degree': [2, 3, 4],  # Only relevant for 'poly' kernel
        'coef0': [0.0, 0.1, 0.5],  # Only relevant for 'poly' and 'sigmoid' kernels
        'C': [0.5, 1, 5, 10, 50, 100]
    }
    param_grid_lasso = {
            'alpha':[0.0001, 0.001, 0.01, 0.1, 1.0]
    }
    param_grid_ridge = {
            'alpha':[0.0001, 0.001, 0.01, 0.1, 1.0]
    }

    # #################################################################
    # --------------------- END OF USER INPUT ---------------------
    # #################################################################

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

    x = x_dataframe.to_numpy(dtype=np.float32)
    y = y_dataframe[target].to_numpy(dtype=np.float32)
    y_binned = np.digitize(y, np.quantile(y, defined_qs))

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    # kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    split_idxs_all_folds = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y_binned)):
    # for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
        split_idxs_all_folds.append((train_idx, val_idx))
        ks_stat, p_value = ks_2samp(y[train_idx], y[val_idx])
        print(f'Fold {fold} : KS statistic = {ks_stat}, p-value = {p_value}')

    grid = GridSearchCV(estimator=XGBRegressor(random_state=random_state),
                        param_grid=param_grid_xgb,
                        scoring='neg_mean_absolute_error', 
                        cv=split_idxs_all_folds, # If using None defaults to 5 fold cross-validation
                        n_jobs=-1, 
                        verbose=1,
                        return_train_score=True)
    grid.fit(x, y.squeeze())
    print(f'Best parameters found: {grid.best_params_}') # Parameter setting that gave best performance on hold out data
    print(f'Best Cross validated score of the best estimator found: {-grid.best_score_}') # Mean cross-validated score of the best_estimator
    # Get the mean absolute error on the training set across all folds for best model
    print(f"Mean test score of the best estimator across all folds : {grid.cv_results_['mean_test_score'][grid.best_index_]}")
    print(f"Std of test score of the best estimator across all folds : {grid.cv_results_['std_test_score'][grid.best_index_]}")
    print(f"Mean train score of the best estimator across all folds : {grid.cv_results_['mean_train_score'][grid.best_index_]}")
    print(f"Std of train score of the best estimator across all folds : {grid.cv_results_['std_train_score'][grid.best_index_]}")

    # Find the indices of top 10 test scores
    top_10_indices = sorted(range(len(grid.cv_results_['mean_test_score'])), key=lambda i: grid.cv_results_['mean_test_score'][i], reverse=True)[:10]
    # Get the top 10 parameter sets
    print("Top 10 parameter sets:")
    for idx in top_10_indices:
        print(f"Params: {grid.cv_results_['params'][idx]}, Mean Test Score: {grid.cv_results_['mean_test_score'][idx]}, Std Test Score: {grid.cv_results_['std_test_score'][idx]}")

    # model_name =     ['XgBoost', 'RandomForest', 'NuSVR', 'Lasso', 'Ridge', 'NestedAE', 'NestedHD']
    # mean_mae_test =  [ 0.127, 0.149, 0.146, 0.235, 0.235, 0.150, 0.150]
    # std_mae_test =   [ 0.026, 0.034, 0.034, 0.028, 0.028, 0.030, 0.030]
    # mean_mae_train = [ 0.026, 0.058, 0.095, 0.229, 0.229, 0.090, 0.090]
    # std_mae_train =  [ 0.001, 0.001, 0.004, 0.003, 0.003, 0.000, 0.010]

    # # Scatter plot true and predicted bandgaps
    # fig, ax = plt.subplots(figsize=(3.5, 3.0))
    # # Create plots to show mean and standard deviation of MAE for test and train datasets
    # ax.errorbar(model_name, y=mean_mae_train, yerr=std_mae_train, label='Train', fmt='o', color='blue', capsize=5)
    # ax.errorbar(model_name, y=mean_mae_test, yerr=std_mae_test, label='Val', fmt='o', color='orange', capsize=5)
    # ax.set_xticks(np.arange(len(model_name)))
    # ax.set_xticklabels(model_name, rotation=45, ha='right')
    # ax.set_ylabel('Bandgap MAE (eV)')
    # plt.tight_layout()
    # plt.legend(frameon=False, fontsize=8)
    # plt.savefig('NAE_v_NHD_v_other_ML_mhp_bandgaps.pdf', bbox_inches='tight', dpi=300)