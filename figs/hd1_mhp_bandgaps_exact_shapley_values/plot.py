import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from shap.explainers import Exact
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ks_2samp

from model import AE, Arctanh

atanh_act_fn = Arctanh()

class OveridedExactExplainer(Exact):
    def __call__(self, *args, **kwargs):
        kwargs.setdefault("max_evals", 150000)  # your new default
        return super().__call__(*args, **kwargs)

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
    # Autoencoder 1 Parameters
    #################################################################

    model_dir = '../../runs/hd1_bandgaps_THEN_perov_solv_BE/fold2_soft_constraints_10D'
    latent_dim = 10
    fold_num = 2
    module_params = {'name':'AE1',
                            'modules':{
                                
                                'encoder':{
                                    'input_dim':15,
                                    'output_dim':latent_dim, 
                                    'hidden_dim':25,
                                    'hidden_layers':1, 
                                    'hidden_activation':None, 
                                    'output_activation':torch.nn.Tanh(), 
                                    'layer_kernel_init':'xavier_normal', 
                                    'layer_bias_init':'zeros'},


                                'bandgaps_predictor':{
                                    'input_dim':latent_dim,
                                    'output_dim':1,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':torch.nn.ReLU(),
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'},

                                'A_predictor':{
                                    'input_dim':latent_dim,
                                    'output_dim':5,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':None,
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'
                                },

                                'B_predictor':{
                                    'input_dim':latent_dim,
                                    'output_dim':6,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':None,
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'
                                },

                                'X_predictor':{
                                    'input_dim':latent_dim,
                                    'output_dim':3,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':None,
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'
                                }

                    }}

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

    #################################################################
    # Script Parameters
    #################################################################

    features_for_SHAP_calc = 'descriptors' # 'descriptors' or 'latents'
    target_for_SHAP_calc = 'bg' # 'bg' or 'latent'
    which_latent_dim = 10  # Only used if target_for_SHAP_calc is 'latent'
    which_dataset_to_use = 'val'  # 'train' or 'val'
    rename_columns = None
    load_data = False
    data_filename = 'shapley_vals_for_bg_pred_using_fold2_val.csv'

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

        print('(INFO) Using a stratified k fold split strategy.')
        train_idxs = []
        val_idxs = []
        y = y_dataframe[target].to_numpy(dtype=np.float32)
        skf = StratifiedKFold(n_splits=int(1/(1 - train_split)), shuffle=True, random_state=random_state)
        y_binned = np.digitize(y, np.quantile(y, defined_qs))
        for fold, (train_idx, val_idx) in enumerate(skf.split(x_dataframe.to_numpy(dtype=np.float32), y_binned)):
            train_idxs.append(train_idx)
            val_idxs.append(val_idx)
            ks_stat, p_val = ks_2samp(y[train_idx], y[val_idx])
            print(f'(INFO) Fold {fold} ks-stat for target: {np.round(ks_stat, 3)}, p-value: {np.round(p_val, 3)}')

        train_dataset = x_dataframe.to_numpy(dtype=np.float32)[train_idxs[fold_num]]
        val_dataset = x_dataframe.to_numpy(dtype=np.float32)[val_idxs[fold_num]]
        train_dataset_df = pd.DataFrame(train_dataset[:, :len(descriptors)], columns=descriptors)
        val_dataset_df = pd.DataFrame(val_dataset[:, :len(descriptors)], columns=descriptors)

        # Load the nested autoencoder model
        loaded_model = AE(module_params)
        loaded_model.load_state_dict(torch.load(model_dir))

        if features_for_SHAP_calc == 'latents':
            if which_dataset_to_use == 'train':
                loaded_model.eval()
                with torch.no_grad():
                    train_tensors = torch.from_numpy(train_dataset_df.values).to(dtype=torch.float32)
                    for i, layer in enumerate(loaded_model.ae_modules['encoder']):
                        if i == 0:
                            train_latents = layer(train_tensors)
                        else:
                            train_latents = layer(train_latents)
                train_latents_df = pd.DataFrame(train_latents.numpy(), columns=[f'latent_{i+1}' for i in range(train_latents.shape[1])])

            if which_dataset_to_use == 'val':
                loaded_model.eval()
                with torch.no_grad():
                    val_tensors = torch.from_numpy(val_dataset_df.values).to(dtype=torch.float32)
                    for i, layer in enumerate(loaded_model.ae_modules['encoder']):
                        if i == 0:
                            val_latents = layer(val_tensors)
                        else:
                            val_latents = layer(val_latents)
                val_latents_df = pd.DataFrame(val_latents.numpy(), columns=[f'latent_{i+1}' for i in range(val_latents.shape[1])])

        if target_for_SHAP_calc == 'bg':
            print(f'(INFO) Calculating SHAP values for BG prediction')
            if features_for_SHAP_calc == 'descriptors':
                print(f'(INFO) Using descriptors as features for SHAP calculation')
                def convert_to_tensor(obs):
                    obs_tensor = torch.from_numpy(obs.values).to(dtype=torch.float32)
                    loaded_model.eval()
                    with torch.no_grad():
                        for i, layer in enumerate(loaded_model.ae_modules['encoder']):
                            if i == 0:
                                z = layer(obs_tensor)
                            else:
                                z = layer(z)
                        for i, layer in enumerate(loaded_model.ae_modules['bandgaps_predictor']):
                            if i == 0:
                                bg_pred = layer(z)
                            else:
                                bg_pred = layer(bg_pred)
                    return bg_pred

                if which_dataset_to_use == 'train':
                    print(f'(INFO) Using training dataset for SHAP calculation')
                    explainer = OveridedExactExplainer(convert_to_tensor, train_dataset_df)
                    shapley_values = explainer(train_dataset_df)
                    columns = train_dataset_df.columns
                
                if which_dataset_to_use == 'val':
                    print(f'(INFO) Using validation dataset for SHAP calculation')
                    explainer = OveridedExactExplainer(convert_to_tensor, val_dataset_df)
                    shapley_values = explainer(val_dataset_df)
                    columns = val_dataset_df.columns

            if features_for_SHAP_calc == 'latents':
                print(f'(INFO) Using latents as features for SHAP calculation')
                def convert_to_tensor(obs):
                    obs_tensor = torch.from_numpy(obs.values).to(dtype=torch.float32)
                    loaded_model.eval()
                    with torch.no_grad():
                        for i, layer in enumerate(loaded_model.ae_modules['bandgaps_predictor']):
                            if i == 0:
                                bg_pred = layer(obs_tensor)
                            else:
                                bg_pred = layer(bg_pred)
                    return bg_pred
                
                if which_dataset_to_use == 'train':
                    print(f'(INFO) Using training dataset for SHAP calculation')
                    explainer = OveridedExactExplainer(convert_to_tensor, train_latents_df)
                    shapley_values = explainer(train_latents_df)
                    columns = train_latents_df.columns

                if which_dataset_to_use == 'val':
                    print(f'(INFO) Using validation dataset for SHAP calculation')
                    explainer = OveridedExactExplainer(convert_to_tensor, val_latents_df)
                    shapley_values = explainer(val_latents_df)
                    columns = val_latents_df.columns

        if target_for_SHAP_calc == 'latent':
            print(f'(INFO) Calculating SHAP values for latent dimension {which_latent_dim}')
            def convert_to_tensor(obs):
                obs_tensor = torch.from_numpy(obs.values).to(dtype=torch.float32)
                loaded_model.eval()
                with torch.no_grad():
                    for i, layer in enumerate(loaded_model.ae_modules['encoder']):
                        if i == 0:
                            z = layer(obs_tensor)
                        else:
                            z = layer(z)
                return z[:, which_latent_dim]

            if which_dataset_to_use == 'train':
                print(f'(INFO) Using training dataset for SHAP calculation')
                explainer = OveridedExactExplainer(convert_to_tensor, train_dataset_df)
                shapley_values = explainer(train_dataset_df)
                columns = train_dataset_df.columns

            if which_dataset_to_use == 'val':
                print(f'(INFO) Using validation dataset for SHAP calculation')
                explainer = OveridedExactExplainer(convert_to_tensor, val_dataset_df)
                shapley_values = explainer(val_dataset_df)
                columns = val_dataset_df.columns

        # Save the shapley values to csv
        shapley_values_df = pd.DataFrame(shapley_values.values, columns=columns)
        if target_for_SHAP_calc == 'latent':
            shapley_values_df.to_csv(f'shapley_vals_for_l{which_latent_dim}_pred_using_fold{fold_num}_{which_dataset_to_use}.csv', index=False)
        else:
            shapley_values_df.to_csv(f'shapley_vals_for_{target_for_SHAP_calc}_pred_using_fold{fold_num}_{which_dataset_to_use}.csv', index=False)
    else:
        shapley_values_df = pd.read_csv(data_filename)

    if rename_columns is not None:
        shapley_values_df.columns = rename_columns

    mean_abs_shapley_values = []

    for feat in shapley_values_df.columns:
        mean_abs_shapley_values.append(np.mean(np.abs(shapley_values_df[feat])))

    mean_abs_shapley_values, feats_for_plot = zip(*sorted(zip(mean_abs_shapley_values, shapley_values_df.columns), reverse=False))

    fig, ax = plt.subplots(figsize=(4.3, 3))
    ax.barh(feats_for_plot, mean_abs_shapley_values, color='skyblue')
    ax.set_xlabel(r'$ \textrm{Mean} ( | \textrm{SHAP value} | )$')
    ax.set_ylabel(r'Feature')
    ax.set_xticks(np.arange(min(mean_abs_shapley_values), max(mean_abs_shapley_values), 0.1))
    # # Label the horizontal bars with their values
    # for i, v in enumerate(mean_abs_shapley_values):
    #     ax.text(v + 0.005, i, f"{v:.4f}", color='blue', va='center', fontsize=8)
    # # Remove the top and left frame lines
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if target_for_SHAP_calc == 'latent':
        plt.savefig(f'shapley_vals_for_l{which_latent_dim}_pred_using_fold{fold_num}_{which_dataset_to_use}.pdf', bbox_inches='tight', dpi=300)
    else:
        plt.savefig(f'shapley_vals_for_{target_for_SHAP_calc}_pred_using_fold{fold_num}_{which_dataset_to_use}.pdf', bbox_inches='tight', dpi=300)

