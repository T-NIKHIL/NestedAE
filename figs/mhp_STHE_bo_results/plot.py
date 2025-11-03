import copy
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.stats import ks_2samp

from model import AE, Arctanh

atanh_act_fn = Arctanh()

class ExactGPModel(ExactGP, GPyTorchModel):
    _num_outputs = 1
    MIN_INFERRED_NOISE_LEVEL = 1e-5

    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        if kernel == 'RBF':
            self.covar_module = ScaleKernel(RBFKernel())
        elif kernel == 'Matern':
            self.covar_module = ScaleKernel(MaternKernel(nu=2.5))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
def train_model(train_x, train_y, epochs, lr):
    likelihood = GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood, 'Matern')

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    for i in range(epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    return model, likelihood

def optimize_acq_fn_and_get_observation(acq_fn, x_test, y_test):
    candidates, _ = optimize_acqf_discrete(
        acq_function=acq_fn,
        choices=x_test, # IMP : Restricts the space from which to select the next candidate 
        q=1, # Number of candidates to select
        max_batch_size=2048,  
        num_restarts=10, # Number of random restarts for optimizer
        raw_samples=512, # Number of random samples for initialization for optimizer
        options={"batch_limit": 5, "maxiter": 200},
        unique=True # Whether to return only unique candidates
    )

    # observe new values. In this case just retreieve 1 value
    x_new = candidates.detach()
    # Find index where it matches in x_test
    match_idx = (x_test == x_new).all(dim = 1)
    # Find where it matches in y_test
    y_new = y_test[match_idx]

    # Remove from the candidate from the test
    x_test = x_test[~match_idx]
    y_test = y_test[~match_idx]

    return x_new, y_new, match_idx, x_test, y_test

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
    # Autoencoder 2 Parameters
    #################################################################

    model_dir_nestedae = '../../runs/nestedae_ae2_perov_STHE/fold2_10D'
    latent_dim_nestedae = 10
    fold_num_nestedae = 2
    module_params_nestedae = {'name':'AE1', 
                        'modules':{

                            'encoder':{
                                'input_dim':14,
                                'output_dim':latent_dim_nestedae, 
                                'hidden_dim':25, 
                                'hidden_layers':1, 
                                'hidden_activation':None, 
                                'output_activation':torch.nn.Tanh(), 
                                'layer_kernel_init':'xavier_normal', 
                                'layer_bias_init':'zeros'},

                            'STHE_predictor':{
                                'input_dim':latent_dim_nestedae,
                                'output_dim':1,
                                'hidden_dim':25,
                                'hidden_layers':1,
                                'hidden_activation':None,
                                'output_activation':torch.nn.ReLU(),
                                'layer_kernel_init':'xavier_normal',
                                'layer_bias_init':'zeros'},

                            'phase_predictor':{
                                'input_dim':latent_dim_nestedae,
                                'output_dim':4,
                                'hidden_dim':25,
                                'hidden_layers':1,
                                'hidden_activation':torch.nn.ReLU(),
                                'output_activation':None,
                                'layer_kernel_init':'xavier_normal',
                                'layer_bias_init':'zeros'},    
                            
                            'latents_predictor':{
                                'input_dim':latent_dim_nestedae,
                                'output_dim':7,
                                'hidden_dim':25,
                                'hidden_layers':1,
                                'hidden_activation':None,
                                'output_activation':None,
                                'layer_kernel_init':'xavier_normal',
                                'layer_bias_init':'zeros'},   

                            'decoder':{
                                'input_dim':latent_dim_nestedae,
                                'output_dim':7,
                                'hidden_dim':25,
                                'hidden_layers':1,
                                'hidden_activation':None,
                                'output_activation':None,
                                'layer_kernel_init':'xavier_normal',
                                'layer_bias_init':'zeros'
                            }

                        }}

    #################################################################
    # SingleAE Parameters
    #################################################################

    model_dir_singleae = '../../runs/singleae_perov_STHE/fold6_22D'
    latent_dim_singleae = 22
    fold_num_singleae = 6
    module_params_singleae = {'name':'AE1', 
                        'modules':{

                        'encoder':{
                              'input_dim':26,
                              'output_dim':latent_dim_singleae, 
                              'hidden_dim':50, 
                              'hidden_layers':1, 
                              'hidden_activation':None, 
                              'output_activation':torch.nn.Tanh(), 
                              'layer_kernel_init':'xavier_normal', 
                              'layer_bias_init':'zeros'},

                          'STHE_predictor':{
                              'input_dim':latent_dim_singleae,
                              'output_dim':1,
                              'hidden_dim':50,
                              'hidden_layers':1,
                              'hidden_activation':torch.nn.ReLU(),
                              'output_activation':torch.nn.ReLU(),
                              'layer_kernel_init':'xavier_normal',
                              'layer_bias_init':'zeros'},

                          'A_predictor':{
                            'input_dim':latent_dim_singleae,
                            'output_dim':4,
                            'hidden_dim':50,
                            'hidden_layers':1,
                            'hidden_activation':torch.nn.ReLU(),
                            'output_activation':None, # Logits
                            'layer_kernel_init':'xavier_normal',
                            'layer_bias_init':'zeros'},
                            

                        'B_predictor':{
                            'input_dim':latent_dim_singleae,
                            'output_dim':6,
                            'hidden_dim':50,
                            'hidden_layers':1,
                            'hidden_activation':torch.nn.ReLU(),
                            'output_activation':None, # Logits
                            'layer_kernel_init':'xavier_normal',
                            'layer_bias_init':'zeros'},

                        'X_predictor':{
                            'input_dim':latent_dim_singleae,
                            'output_dim':3,
                            'hidden_dim':50,
                            'hidden_layers':1,
                            'hidden_activation':torch.nn.ReLU(),
                            'output_activation':None, # Logits
                            'layer_kernel_init':'xavier_normal',
                            'layer_bias_init':'zeros'},  

                          'phase_predictor':{
                              'input_dim':latent_dim_singleae,
                              'output_dim':4,
                              'hidden_dim':50,
                              'hidden_layers':1,
                              'hidden_activation':torch.nn.ReLU(),
                              'output_activation':None,
                              'layer_kernel_init':'xavier_normal',
                              'layer_bias_init':'zeros'}, 

                          'decoder':{
                              'input_dim':latent_dim_singleae,
                              'output_dim':26,
                              'hidden_dim':50,
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

    dataset_loc = '../../datasets/MHP_for_water_splitting/dataset_with_ae1_latents.csv'

    # Compositional Factors
    # latent_col_names = ['l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6']
    latent_col_names = []

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
                   'X_AN',
                   'A_En_mull',
                   'B_En_mull',
                   'X_En_mull',
                   'x(S)', # Geometric mean of mulliken electronegativities
    # Structural + Electronic factors
    # descriptors = [
                   't',
                   'o',
                   'tao',
                   'CBM',
                   'VBM',
                   'n_abs(%)', # Efficiency of light absorption
                   'n_cu(%)'] # Efficiency of carrier utilization

    target = ['n_STH(%)']

    standardize_descs = True
    defined_qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    #################################################################
    # Script Parameters
    #################################################################

    skip_duplicate_latent_check = False
    nestedae_or_nestedhd_feat_sel = 'singleae' # None or 'nestedae' or 'singleae'
    test_size = 0.996
    bo_n_trials = 200
    bo_n_updates = 10
    bo_n_batches = 200
    epochs_train_GP = 100
    lr_GP = 0.1
    maximize = True
    verbose = True
    # Plot parameters
    y_axis_tick_range = np.arange(0, 10, 1)
    show_mean = False
    show_median = False

    load_data = True
    which_files_to_load = ['best_obs_val_all_trials_tensor_SimpleBO.pt',
                           'best_obs_val_all_trials_tensor_SingleAE.pt',
                           'best_obs_val_all_trials_tensor_NestedAE.pt']

    save_fig_filename = 'space_explored_comparison_max.pdf'

    # #################################################################
    # --------------------- END OF USER INPUT ---------------------
    # #################################################################

    if not load_data:
        x_dataframe = pd.read_csv(dataset_loc)[descriptors + latent_col_names]
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

        x_torch_all = torch.from_numpy(x_dataframe.to_numpy()[:, 0:(len(descriptors + latent_col_names))]).to(dtype=torch.float32)
        y_true = torch.from_numpy(y_dataframe.to_numpy()).to(dtype=torch.float32)
        total_number_of_materials_in_dataset = x_torch_all.shape[0]

        # Check for nan in dataset
        if np.isnan(x_torch_all).any():
            print('(INFO) Dataset has NaN values. Please check.')
        else:
            print('(INFO) Dataset has no NaN values.')

        if nestedae_or_nestedhd_feat_sel == 'nestedae':
            loaded_model = AE(module_params_nestedae)
            loaded_model.load_state_dict(torch.load(model_dir_nestedae))
            loaded_model.eval()
            with torch.no_grad():
                for i, layer in enumerate(loaded_model.ae_modules['encoder']):
                    if i == 0:
                        latents = layer(x_torch_all)
                    else:
                        latents = layer(latents)
            latents_mean_dict = {}
            latents_std_dev_dict = {}
            desc_stand = copy.deepcopy(latents)
            for dim in range(latent_dim_nestedae):
                mean = latents[:, dim].mean()
                latents_mean_dict[dim] = mean
                std_dev = latents[:, dim].std()
                latents_std_dev_dict[dim] = std_dev
                desc_stand[:, dim] = (latents[:, dim] - mean) / std_dev
        elif nestedae_or_nestedhd_feat_sel == 'singleae':
            loaded_model = AE(module_params_singleae)
            loaded_model.load_state_dict(torch.load(model_dir_singleae))
            loaded_model.eval()
            with torch.no_grad():
                for i, layer in enumerate(loaded_model.ae_modules['encoder']):
                    if i == 0:
                        latents = layer(x_torch_all)
                    else:
                        latents = layer(latents)
            latents_mean_dict = {}
            latents_std_dev_dict = {}
            desc_stand = copy.deepcopy(latents)
            for dim in range(latent_dim_singleae):
                mean = latents[:, dim].mean()
                latents_mean_dict[dim] = mean
                std_dev = latents[:, dim].std()
                latents_std_dev_dict[dim] = std_dev
                desc_stand[:, dim] = (latents[:, dim] - mean) / std_dev
        else:
            desc_stand = x_torch_all

        # Sanity check : Identify row indices where the latent values are the same in latents
        if not skip_duplicate_latent_check:
            duplicate_latent_indices = []
            for i in range(desc_stand.shape[0]):
                for j in range(i + 1, desc_stand.shape[0]):
                    if torch.equal(desc_stand[i], desc_stand[j]):
                        duplicate_latent_indices.append((i, j))
            if len(duplicate_latent_indices) > 0:
                print(f'(INFO) Found duplicate latent representations at indices: {duplicate_latent_indices}')
                duplicate_latent_1 = desc_stand[duplicate_latent_indices[0][0], :].detach().numpy()
                duplicate_latent_2 = desc_stand[duplicate_latent_indices[0][1], :].detach().numpy()
                duplicate_latent_1_unstand = np.zeros(duplicate_latent_1.shape)
                duplicate_latent_2_unstand = np.zeros(duplicate_latent_2.shape)
                for i, desc in enumerate(x_dataframe.columns.tolist()):
                    duplicate_latent_1_unstand[i] = duplicate_latent_1[i] * desc_std_devs[desc] + desc_means_dict[desc]
                    duplicate_latent_2_unstand[i] = duplicate_latent_2[i] * desc_std_devs[desc] + desc_means_dict[desc]
                print(duplicate_latent_1_unstand)
                print(duplicate_latent_2_unstand)
                raise ValueError("Duplicate latent representations found.")

        best_obs_val_all_trials = []
        selected_desc_all_trials = []
        selected_obs_all_trials = []

        trial_num = 1
        while True:
            best_obs_val_each_trial = []
            selected_desc_each_trial = []
            selected_obs_each_trial = []

            t0 = time.monotonic()

            print(f'(INFO) ----- Trial {trial_num} ----- ')
            y_true_binned = np.digitize(y_true.numpy(), np.quantile(y_true.numpy(), defined_qs))
            desc_train_stand, desc_test_stand, y_train, y_test = train_test_split(desc_stand, y_true, 
                                                                                  test_size=test_size,
                                                                                  random_state=trial_num)
            print(f'(INFO) Train size: {desc_train_stand.shape[0]}, Test size: {desc_test_stand.shape[0]}')
            ks_stat, p_val = ks_2samp(y_train.numpy(), y_test.numpy())
            print(f'(INFO) KS-stat for target between train and test: {np.round(ks_stat, 3)}, p-value: {np.round(p_val, 3)}')
            initial_train_size = desc_train_stand.shape[0]

            # check for max in preds
            if maximize : 
                best_obs_val = y_train.max()
                best_obs_val_idx = y_train.argmax()
                optimal_soln = torch.cat([y_train, y_test]).max()
            else:
                best_obs_val = y_train.min()
                best_obs_val_idx = y_train.argmin()
                optimal_soln = torch.cat([y_train, y_test]).min()

            if best_obs_val.eq(optimal_soln):
                if maximize:
                    print(f'(INFO) Max in training set. Removing it. Max value is {optimal_soln}')
                else:
                    print(f'(INFO) Min in training set. Removing it. Min value is {optimal_soln}')
                # removing from train and adding to test
                desc_test_stand = torch.cat([desc_test_stand, desc_train_stand[best_obs_val_idx, :].unsqueeze(0)], dim=0)
                y_test = torch.cat([y_test, y_train[best_obs_val_idx].unsqueeze(0)], dim=0)
                desc_train_stand = torch.cat([desc_train_stand[0:best_obs_val_idx, :], desc_train_stand[best_obs_val_idx + 1:, :]], dim=0)
                y_train = torch.cat([y_train[0:best_obs_val_idx], y_train[best_obs_val_idx + 1:]], dim=0)
                if maximize:
                    # Updating best obs value
                    print('(INFO) Updating best observed value')
                    best_obs_val = y_train.max()
                else:
                    print('(INFO) Updating best observed value')
                    best_obs_val = y_train.min()
            else:
                if maximize:
                    print(f'(INFO) Max not located in training set. Proceeding with BO')
                else:
                    print(f'(INFO) Min not located in training set. Proceeding with BO')

            # Add the initial best obs val in training set to trial history
            best_obs_val_each_trial.append(best_obs_val)
            print(f'(INFO) Initial best observed value in training set: {best_obs_val}')

            likelihood = GaussianLikelihood()
            model = ExactGPModel(desc_train_stand, y_train.squeeze(), likelihood, 'Matern')
            acq_fn = ExpectedImprovement(model=model, best_f=best_obs_val_each_trial[-1], maximize=maximize)

            for batch_num in range(1, bo_n_batches + 1):
                # Train the model on the initial training data. Then update it with points chosen by the acq. fn from the test set
                if (batch_num - 1)%bo_n_updates == 0:
                    if verbose: print(f'Updating the model.')
                    model, likelihood = train_model(desc_train_stand, y_train.squeeze(), epochs_train_GP, lr_GP)

                desc_new, y_new, match_idx, desc_test_stand, y_test = optimize_acq_fn_and_get_observation(acq_fn, desc_test_stand, y_test)
        
                # Add the latent and observation to training set.
                desc_train_stand = torch.cat([desc_train_stand, desc_new], dim=0)
                y_train = torch.cat([y_train, y_new], dim=0)

                # Track the latent and corresponding obs
                selected_desc_each_trial.append(desc_new)
                selected_obs_each_trial.append(y_new)

                if maximize:
                    best_obs_val_each_trial.append(y_train.max())
                else:
                    best_obs_val_each_trial.append(y_train.min())

                # Define acquisition function with updated GP model and best observed obs so far.
                acq_fn = ExpectedImprovement(model=model, best_f=best_obs_val_each_trial[-1], maximize=maximize)

                if verbose:
                    print(f'Batch {batch_num}, best obs val {best_obs_val_each_trial[-1]}')

            t1 = time.monotonic()
            print(f' Time = {round(t1 - t0, 3)}')
            if best_obs_val_each_trial[-1] != optimal_soln:
                print('(WARNING) Optimal solution not found in this trial.')
                with open('optimal_not_found_runs.txt', 'a') as f:
                    f.write(f'Trial {trial_num} : Best obs val found: {best_obs_val_each_trial[-1]}, Initial Best : {best_obs_val_each_trial[0]}, Optimal soln: {optimal_soln}\n')
                bo_n_trials += 1
            else:
                best_obs_val_each_trial_tensor = torch.stack(best_obs_val_each_trial, dim=0)
                selected_desc_each_trial_tensor = torch.stack([desc.squeeze() for desc in selected_desc_each_trial], dim=0)
                selected_obs_each_trial_tensor = torch.stack([obs.squeeze() for obs in selected_obs_each_trial], dim=0)
                best_obs_val_all_trials.append(best_obs_val_each_trial_tensor)
                selected_desc_all_trials.append(selected_desc_each_trial_tensor)
                selected_obs_all_trials.append(selected_obs_each_trial_tensor)

            trial_num += 1
            if trial_num > bo_n_trials:
                break

        # Save the lists to numpy arrays
        if nestedae_or_nestedhd_feat_sel is None:
            list_of_methods = ['SimpleBO']
        elif nestedae_or_nestedhd_feat_sel == 'nestedae':
            list_of_methods = ['NestedAE']
        else:
            list_of_methods = ['SingleAE']
        best_obs_val_all_trials_tensor = torch.stack(best_obs_val_all_trials, dim=0)
        selected_desc_all_trials_tensor = torch.stack(selected_desc_all_trials, dim=0)
        selected_obs_all_trials_tensor = torch.stack(selected_obs_all_trials, dim=0)
        torch.save(best_obs_val_all_trials_tensor, f'best_obs_val_all_trials_tensor_{list_of_methods[-1]}.pt')
        torch.save(selected_desc_all_trials_tensor, f'selected_desc_all_trials_tensor_{list_of_methods[-1]}.pt')
        torch.save(selected_obs_all_trials_tensor, f'selected_obs_all_trials_tensor_{list_of_methods[-1]}.pt')
        list_of_best_obs_all_trials_tensor = [best_obs_val_all_trials_tensor]
    else:
        list_of_best_obs_all_trials_tensor = []
        list_of_methods = []
        for which_file in which_files_to_load:
            list_of_best_obs_all_trials_tensor.append(torch.load(which_file))
            list_of_methods.append(which_file.split('_')[-1].split('.')[0]) # Extract method name from filename
        
        total_number_of_materials_in_dataset = pd.read_csv(dataset_loc)[descriptors + latent_col_names].to_numpy().shape[0]
        initial_train_size = (1 - test_size)*total_number_of_materials_in_dataset

    # overlay the 100 trials in line plot with varying shades of blue
    # plt.figure(figsize=(8, 6))
    # for i in range(len(best_obs_val_hist)):
    #     plt.plot(range(1, len(best_obs_val_hist[i]) + 1), best_obs_val_hist[i], color='blue', alpha=0.1)
    #     # Mark the initial best observed value
    #     plt.scatter(1, best_obs_val_hist[i][0], color='red', s=10)

    # best_obs_val_hist_arr = np.array(best_obs_val_hist)
    # print(np.mean(best_obs_val_hist_arr, axis=0).shape)
    # plt.plot(range(1, len(best_obs_val_hist_arr[0]) + 1), np.mean(best_obs_val_hist_arr, axis=0), color='orange', label='Mean of 200 trials', linewidth=2)
    # plt.legend()

    # # for i in range(len(selected_obs_hist)):
    # #     selected_obs_concat = torch.cat(selected_obs_hist[i]).numpy()
    # #     plt.scatter(range(1, len(selected_obs_concat) + 1), selected_obs_concat, color='blue', s=10, alpha=0.1)
    # plt.xlabel('Number of evaluated design candidates', fontsize=14)
    # plt.ylabel('Best Observed Value', fontsize=14)
    # plt.grid()

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    # Percent Space Explored Plot
    text_pos = [0.0, 1.0, 2.0]
    k = 0
    # Percent Space Explored Plot
    for method, best_obs_all_trials in zip(list_of_methods, list_of_best_obs_all_trials_tensor):
        percent_space_splored = []
        for trial in range(bo_n_trials):
            if maximize:
                # Need to add the initial training set size. 
                # Another point is that we add the best obs val from the initial training set. So the first entry is not actually selected by the acq. fn.
                # This is fine since anyways argmax index counting starts from 0.
                percent_space_splored.append((initial_train_size + best_obs_all_trials[trial, :].argmax().item())/total_number_of_materials_in_dataset*100)
            else:
                percent_space_splored.append((initial_train_size + best_obs_all_trials[trial, :].argmin().item())/total_number_of_materials_in_dataset*100)
            
        # Plot box plot and strip plot of iter_to_reach_max
        percent_space_explored_df = pd.DataFrame(percent_space_splored, columns=['value'])
        percent_space_explored_df['method'] = method
        percent_space_explored_df.to_csv(f'percent_space_explored_{method}.csv', index=False)

        ax = sns.boxplot(x='method', y='value', data=percent_space_explored_df, 
                        showfliers=True, width=0.5,
                        medianprops=dict(color='black', linewidth=1.0))
        # This is to display the dots 
        ax = sns.stripplot(x='method', y='value', data=percent_space_explored_df, 
                        hue='method', legend=False,
                        alpha=.4, linewidth=1, jitter=0.1, size=1)
        mean_val = percent_space_explored_df['value'].mean()
        median_val = percent_space_explored_df['value'].median()
        ax.text(text_pos[k]-0.38, median_val - 0.1, f"{median_val:.1f}", ha='center', fontsize=8, color='black', fontweight='bold')
        if show_mean:
            # Show mean as a horizontal dotted red line
            ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label='Mean')
            ax.text(-0.4, mean_val + 0.6, f"Mean: {mean_val:.1f}", ha='center', fontsize=8, color='black', fontweight='bold')
        if show_median:
            # Show median as a horizontal dotted black line
            ax.axhline(median_val, color='black', linestyle='--', linewidth=2, label='Median')
        # Set y axis values
        ax.set_yticks(y_axis_tick_range)
        k += 1

    ax.set_xticks(list_of_methods)
    ax.set_xticklabels(list_of_methods, rotation=0)
    plt.gca().set_ylabel(r'\% space explored')
    plt.tight_layout()
    plt.savefig(save_fig_filename, bbox_inches='tight', dpi=300)