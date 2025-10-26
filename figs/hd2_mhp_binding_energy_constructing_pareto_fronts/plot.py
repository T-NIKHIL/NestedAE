import copy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from scipy.stats import truncnorm

from model import AE, Arctanh

class Arctanh(torch.nn.Module):
    def forward(self, x):
        return torch.atanh(torch.clamp(x, -0.999999, 0.999999))
    
atanh_act_fn = Arctanh()

def call_loaded_hd1(latents):
    loaded_hd1.eval()
    with torch.no_grad():
        for i, layer in enumerate(loaded_hd1.ae_modules['A_predictor']):
            if i == 0:
                a_pred = layer(latents)
            else:
                a_pred = layer(a_pred)

        for i, layer in enumerate(loaded_hd1.ae_modules['B_predictor']):
            if i == 0:
                b_pred = layer(latents)
            else:
                b_pred = layer(b_pred)

        for i, layer in enumerate(loaded_hd1.ae_modules['X_predictor']):
            if i == 0:
                x_pred = layer(latents)
            else:
                x_pred = layer(x_pred)

        for i, layer in enumerate(loaded_hd1.ae_modules['bandgaps_predictor']):
            if i == 0:
                bg_pred = layer(latents)
            else:
                bg_pred = layer(bg_pred)
    return a_pred, b_pred, x_pred, bg_pred

def call_loaded_hd2(inputs, input_type='latents'):
    if input_type == 'descriptors':
        loaded_hd2.eval()
        with torch.no_grad():
            for i, layer in enumerate(loaded_hd2.ae_modules['encoder']):
                inputs = layer(inputs)
        return inputs
    else:
        loaded_hd2.eval()
        with torch.no_grad():
            for i, layer in enumerate(loaded_hd2.ae_modules['BE_predictor']):
                if i == 0: 
                    be_pred = layer(inputs)
                else: 
                    be_pred = layer(be_pred)

            for i, layer in enumerate(loaded_hd2.ae_modules['latents_predictor']):
                if i == 0: 
                    hd1_latent_pred = layer(inputs)
                else: 
                    hd1_latent_pred = layer(hd1_latent_pred)

            for i, layer in enumerate(loaded_hd2.ae_modules['solvent_predictor']):
                if i == 0: 
                    solvent_pred = layer(inputs)
                else: 
                    solvent_pred = layer(solvent_pred)
        return be_pred, hd1_latent_pred, solvent_pred

if __name__ == '__main__':
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
    # Heterodecoder 2 Parameters
    #################################################################

    model_dir_for_hd2 = '../../runs/hd2_bandgaps_THEN_perov_solv_BE/fold7_soft_constraints_6D'
    latent_dim_for_hd2 = 6
    fold_num_for_hd2 = 7
    module_params_for_hd2 = {'name':'HD2', 
                            'modules':{
                                    
                                'encoder':{
                                    'input_dim':16,
                                    'output_dim':latent_dim_for_hd2, 
                                    'hidden_dim':25,
                                    'hidden_layers':1, 
                                    'hidden_activation':None, 
                                    'output_activation':torch.nn.Tanh(), 
                                    'layer_kernel_init':'xavier_normal', 
                                    'layer_bias_init':'zeros'},

                                'BE_predictor':{
                                    'input_dim':latent_dim_for_hd2,
                                    'output_dim':1,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':torch.nn.ReLU(),
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'},
                                
                                'latents_predictor':{
                                    'input_dim':latent_dim_for_hd2,
                                    'output_dim':10,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':atanh_act_fn,
                                    'output_activation':None,
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'},

                                'solvent_predictor':{
                                    'input_dim':latent_dim_for_hd2,
                                    'output_dim':8,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':None,
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'
                                },

                        }}

    #################################################################
    # Heterodecoder 1 Parameters
    #################################################################

    model_dir_for_hd1 = '../../runs/hd1_bandgaps_THEN_perov_solv_BE/fold2_soft_constraints_10D'
    latent_dim_for_hd1 = 10
    fold_num_for_hd1 = 2
    module_params_for_hd1 = {'name':'HD1',
                            'modules':{
                                
                                'encoder':{
                                    'input_dim':15,
                                    'output_dim':latent_dim_for_hd1, 
                                    'hidden_dim':25,
                                    'hidden_layers':1, 
                                    'hidden_activation':None, 
                                    'output_activation':torch.nn.Tanh(), 
                                    'layer_kernel_init':'xavier_normal', 
                                    'layer_bias_init':'zeros'},

                                'bandgaps_predictor':{
                                    'input_dim':latent_dim_for_hd1,
                                    'output_dim':1,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':torch.nn.ReLU(),
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'},

                                'A_predictor':{
                                    'input_dim':latent_dim_for_hd1,
                                    'output_dim':5,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':None,
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'
                                },

                                'B_predictor':{
                                    'input_dim':latent_dim_for_hd1,
                                    'output_dim':6,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':None,
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'
                                },

                                'X_predictor':{
                                    'input_dim':latent_dim_for_hd1,
                                    'output_dim':3,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':None,
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'
                                }}}

    #################################################################
    # Dataset Parameters
    #################################################################

    dataset_loc_for_hd1 = '../../datasets/MHP_bandgaps_AND_perov_solvent_BE/nestedae_dataset/perov_bandgaps_PBE_arun_reduced.csv'
    dataset_loc_for_hd2 = '../../datasets/MHP_bandgaps_AND_perov_solvent_BE/nestedae_dataset/perov_solv_BE_for_nestedhd.csv'

    descriptors_for_hd1 = ['A_IONRAD',
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
    target_for_hd1 = ['Gap']
    
    descriptors_for_hd2 = ['SOLV_DENSITY',
                           'SOLV_DIELECTRIC',
                           'SOLV_GDN',
                           'SOLV_DPM',
                           'SOLV_MV',
                           'SOLV_UMBO',
                           'l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9']
    latent_descriptors_for_hd2 = ['l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9']
    target_for_hd2 = ['Target']

    standardize_descs = True

    #################################################################
    # Script Parameters
    #################################################################

    selected_latent = None
    maximize_be = True
    which_trial = 1
    sample_method = 'random' # 'linear' or 'trunc_normal' or 'random'
    sample_spacing = 0.01 # Only for linear sampling

    num_samples = 2000 # Only for multivariate normal sampling
    cov = np.diag(np.array([0.08]*latent_dim_for_hd1))

    lower_trunc = -1.0
    upper_trunc = 1.0
    mean_trunc = 0.0
    sigma_trunc = 0.5

    load_data = False
    hd2_results_file = 'predicted_binding_energies_and_designs_for_sampled_hd1_latents.csv'

    # #################################################################
    # --------------------- END OF USER INPUT ---------------------
    # #################################################################

    if not load_data:
        x_dataframe_for_hd2 = pd.read_csv(dataset_loc_for_hd2)[descriptors_for_hd2]

        if standardize_descs:
            hd2_desc_means_dict = {}
            hd2_desc_std_devs = {}
            for desc in x_dataframe_for_hd2.columns.tolist():
                mean = x_dataframe_for_hd2[desc].mean()
                hd2_desc_means_dict[desc] = mean
                std_dev = x_dataframe_for_hd2[desc].std()
                hd2_desc_std_devs[desc] = std_dev
                x_dataframe_for_hd2[desc] = (x_dataframe_for_hd2[desc] - mean) / std_dev
            print('Descriptors standardized.')
        else:
            print('Descriptors not standardized.')

        x_arr_for_hd2 = x_dataframe_for_hd2.to_numpy(dtype=np.float32)

        loaded_hd2 = AE(module_params_for_hd2)
        loaded_hd2.load_state_dict(torch.load(model_dir_for_hd2))

        loaded_hd1 = AE(module_params_for_hd1)
        loaded_hd1.load_state_dict(torch.load(model_dir_for_hd1))

        if selected_latent is None:
            print('(INFO) Using the optimal hd2 latent found from BO as starting point')
            if maximize_be:
                selected_latent_hist = np.load('../hd2_mhp_solv_BE_bo_results/selected_latent_hist_nestedhd_max.npy')
                # removing the first entry as that is the best in training data prior to start of BO
                batches_to_optimal = pd.read_csv('../hd2_mhp_solv_BE_bo_results/bo_results_nestedhd_max.csv')[f'trial{which_trial}'][1:].values
                optimal_latent_idx = np.argmax(batches_to_optimal)
            else:
                selected_latent_hist = np.load('../hd2_mhp_solv_BE_bo_results/selected_latent_hist_nestedhd_min.npy')
                batches_to_optimal = pd.read_csv('../hd2_mhp_solv_BE_bo_results/bo_results_nestedhd_min.csv')[f'trial{which_trial}'][1:].values
                optimal_latent_idx = np.argmin(batches_to_optimal)
            selected_latent = selected_latent_hist[which_trial, optimal_latent_idx, :, :]

            # Unstandardize the selected latent
            loaded_hd2.eval()
            with torch.no_grad():
                x_tensor_for_hd2 = torch.from_numpy(x_arr_for_hd2).to(dtype=torch.float32)
                for i, layer in enumerate(loaded_hd2.ae_modules['encoder']):
                    if i == 0:
                        hd2_latents = layer(x_tensor_for_hd2)
                    else:
                        hd2_latents = layer(hd2_latents)

            hd2_latents_means = hd2_latents.numpy().mean(axis=0)
            hd2_latents_std_devs = hd2_latents.numpy().std(axis=0)
            selected_latent_unstand = (selected_latent * hd2_latents_std_devs) + hd2_latents_means
            selected_latent_tensor = torch.from_numpy(selected_latent_unstand.reshape(1, -1)).to(dtype=torch.float32)
            print(f'(DATA) Found latent with optimal BE of {batches_to_optimal[optimal_latent_idx]} at index {optimal_latent_idx}')
            print(f'(DATA) Selected latent : {selected_latent_tensor.numpy().squeeze()}')
        else:
            selected_latent_tensor = torch.from_numpy(selected_latent.reshape(1, -1)).to(dtype=torch.float32)
            print(f'(DATA) Selected latent : {selected_latent_tensor.numpy().squeeze()}')

        # Get the unique rows in x_dataframe_for_hd2
        unique_stand_solvent_descriptors = x_dataframe_for_hd2[['SOLV_DENSITY',
                                                                'SOLV_DIELECTRIC',
                                                                'SOLV_GDN',
                                                                'SOLV_DPM',
                                                                'SOLV_MV',
                                                                'SOLV_UMBO']].drop_duplicates()

        # # Unstandardize the solvent descriptors
        # unique_stand_solvent_descriptors_unstand = unique_stand_solvent_descriptors.copy()
        # for col in unique_stand_solvent_descriptors.columns:
        #     mean = hd2_desc_means_dict[col]
        #     std_dev = hd2_desc_std_devs[col]
        #     unique_stand_solvent_descriptors_unstand[col] = (unique_stand_solvent_descriptors[col] * std_dev) + mean
        unique_solvent_types_reordered = ['THTO', 'DMSO', 'DMF', 'NMP', 'ACETONE', 'METHA', 'GBL', 'NITRO']
        list_of_solvents = ['DMSO', 'THTO', 'DMF', 'NMP', 'ACETONE', 'METHA', 'GBL', 'NITRO']
        list_of_A_ions = ['K', 'Rb', 'Cs', 'MA', 'FA']
        list_of_B_ions = ['Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb']
        list_of_X_ions = ['Cl', 'Br', 'I']

        be_pred, hd1_latent_pred, solvent_pred = call_loaded_hd2(selected_latent_tensor)
        solvent_pred = list_of_solvents[torch.softmax(solvent_pred, dim=1).numpy().squeeze().argmax()]
        hd2_results_df = pd.DataFrame(selected_latent.reshape(1, -1), columns=[f'HD2_l{i+1}' for i in range(latent_dim_for_hd2)])
        hd2_results_df['Pred_Binding_Energy'] = be_pred.numpy().squeeze()
        hd2_results_df['Pred_Solvent'] = solvent_pred

        print(f'(INFO) Unstandardizing the hd1 latent.')
        hd1_latent_pred_unstand = copy.deepcopy(hd1_latent_pred.squeeze())
        for i, latent_desc in enumerate(latent_descriptors_for_hd2):
            mean = hd2_desc_means_dict[latent_desc]
            std_dev = hd2_desc_std_devs[latent_desc]
            hd1_latent_pred_unstand[i] = (hd1_latent_pred.squeeze()[i] * std_dev) + mean
        hd1_latent_pred_unstand = hd1_latent_pred_unstand.unsqueeze(0)

        # Send the predicted HD1 latents to decoders of HD1
        a_pred, b_pred, x_pred, bg_pred = call_loaded_hd1(hd1_latent_pred_unstand)

        a_pred = torch.softmax(a_pred, dim=1)
        b_pred = torch.softmax(b_pred, dim=1)
        x_pred = torch.softmax(x_pred, dim=1)

        for a_idx, a_ion in enumerate(list_of_A_ions):
            hd2_results_df[f'A_ion_{a_ion}'] = a_pred.numpy().squeeze()[a_idx]
        for b_idx, b_ion in enumerate(list_of_B_ions):
            hd2_results_df[f'B_ion_{b_ion}'] = b_pred.numpy().squeeze()[b_idx]
        for x_idx, x_ion in enumerate(list_of_X_ions):
            hd2_results_df[f'X_ion_{x_ion}'] = x_pred.numpy().squeeze()[x_idx]
        hd2_results_df['Pred_Bandgap'] = bg_pred.numpy().squeeze()
        hd2_results_df.to_csv('predicted_binding_energies_and_designs_for_sampled_hd2_latents.csv', index=False)

        if sample_method == 'linear':
            samples_along_selected_latent_dim = np.arange(-1, 1.1, sample_spacing)
            list_of_new_hd1_latents = []
            for d_idx in range(latent_dim_for_hd1):
                for sample in samples_along_selected_latent_dim:
                    new_hd1_latent = copy.deepcopy(hd1_latent_pred_unstand.squeeze())
                    new_hd1_latent[d_idx] = sample
                    list_of_new_hd1_latents.append(new_hd1_latent)
            tensor_of_new_hd1_latents = torch.stack(list_of_new_hd1_latents, dim=0)
        elif sample_method == 'normal':
            rng = np.random.default_rng(seed=random_state)
            tensor_of_new_hd1_latents = torch.from_numpy(rng.multivariate_normal(mean=np.zeros(latent_dim_for_hd1),
                                                                                cov=cov,
                                                                                size=num_samples)).to(dtype=torch.float32)
        elif sample_method == 'random':
            rng = np.random.default_rng(seed=random_state)
            tensor_of_new_hd1_latents = torch.from_numpy(rng.uniform(low=-1.0,
                                                                     high=1.0,
                                                                     size=(num_samples, latent_dim_for_hd1))).to(dtype=torch.float32)
        elif sample_method == 'trunc_normal':
            rng = np.random.default_rng(seed=random_state)
            trunc_normal_dist = truncnorm((lower_trunc - mean_trunc) / sigma_trunc, (upper_trunc - mean_trunc) / sigma_trunc, loc=mean_trunc, scale=sigma_trunc)
            samples = trunc_normal_dist.rvs((num_samples, latent_dim_for_hd1), random_state=rng)
            tensor_of_new_hd1_latents = torch.from_numpy(samples).to(dtype=torch.float32)
        else:
            raise ValueError(f'Invalid sample_method: {sample_method}')

        print(f'(INFO) Number of sampled latents : {tensor_of_new_hd1_latents.shape[0]}')

        a_pred, b_pred, x_pred, bg_pred = call_loaded_hd1(tensor_of_new_hd1_latents)

        # Send the predictions to a csv file
        hd1_results_df = pd.DataFrame(tensor_of_new_hd1_latents.numpy(), columns=[f'HD1_l{i+1}' for i in range(latent_dim_for_hd1)])
        for a_idx, a_ion in enumerate(list_of_A_ions):
            hd1_results_df[f'A_ion_{a_ion}'] = torch.softmax(a_pred, dim=1).numpy()[:, a_idx]
        for b_idx, b_ion in enumerate(list_of_B_ions):
            hd1_results_df[f'B_ion_{b_ion}'] = torch.softmax(b_pred, dim=1).numpy()[:, b_idx]
        for x_idx, x_ion in enumerate(list_of_X_ions):
            hd1_results_df[f'X_ion_{x_ion}'] = torch.softmax(x_pred, dim=1).numpy()[:, x_idx]
        hd1_results_df['Pred_Bandgap'] = bg_pred.numpy().squeeze()

        # hd1_results_df.to_csv('test.csv', index=False)
        hd1_results_df.to_csv(f'predicted_bandgaps_and_designs_for_sampled_hd1_latents.csv', index=False)

        mhp_design_columns = []
        for a_ion in list_of_A_ions:
            mhp_design_columns.append(f'A_ion_{a_ion}')
        for b_ion in list_of_B_ions:
            mhp_design_columns.append(f'B_ion_{b_ion}')
        for x_ion in list_of_X_ions:
            mhp_design_columns.append(f'X_ion_{x_ion}')

        sampled_hd1_latents = hd1_results_df[[f'HD1_l{i+1}' for i in range(latent_dim_for_hd1)]].to_numpy(dtype=np.float32)
        print('(INFO) Standardizing the sampled hd1 latents.')
        sampled_hd1_latents_stand = copy.deepcopy(sampled_hd1_latents)
        for i, latent_desc in enumerate(latent_descriptors_for_hd2):
            mean = hd2_desc_means_dict[latent_desc]
            std_dev = hd2_desc_std_devs[latent_desc]
            sampled_hd1_latents_stand[i] = (sampled_hd1_latents[i] - mean) / std_dev

        print(f'(INFO) Concatenting the standardized hd1 latent with each of the unique solvent descriptors.')
        list_of_hd2_descriptors = []
        list_of_mhp_designs = []
        list_of_mhp_bg_preds = []
        list_of_true_solvent_types = []
        for (sampled_hd1_latent_stand, mhp_design, mhp_bg_pred) in zip(sampled_hd1_latents_stand, 
                                                                        hd1_results_df[mhp_design_columns].to_numpy(dtype=np.float32),
                                                                        hd1_results_df['Pred_Bandgap'].to_numpy(dtype=np.float32)):
            for (unique_solvent_type, unique_solvent_desc) in zip(unique_solvent_types_reordered, unique_stand_solvent_descriptors.to_numpy(dtype=np.float32)):
                hd2_latent = np.concatenate((unique_solvent_desc, sampled_hd1_latent_stand), axis=0)
                list_of_hd2_descriptors.append(hd2_latent)
                list_of_mhp_designs.append(mhp_design)
                list_of_mhp_bg_preds.append(mhp_bg_pred)
                list_of_true_solvent_types.append(unique_solvent_type)
        tensor_of_hd2_descriptors =  torch.from_numpy(np.array(list_of_hd2_descriptors)).to(dtype=torch.float32)

        print(f'(INFO) Number of hd2 descriptors to evaluate : {tensor_of_hd2_descriptors.shape[0]}')

        hd2_latents_pred = call_loaded_hd2(tensor_of_hd2_descriptors, input_type='descriptors')
        be_pred, hd1_latents_pred, solvent_pred = call_loaded_hd2(hd2_latents_pred)

        hd2_results_df = pd.DataFrame(np.array(list_of_hd2_descriptors)[:, 6:], columns=[f'HD1_l{i+1}' for i in range(latent_dim_for_hd1)])
        hd2_results_df[mhp_design_columns] = np.array(list_of_mhp_designs)
        hd2_results_df['Pred_Bandgap'] = np.array(list_of_mhp_bg_preds)
        hd2_results_df['Pred_Binding_Energy'] = be_pred.numpy().squeeze()
        hd2_results_df['Pred_Solvent'] = [list_of_solvents[torch.softmax(sp.unsqueeze(0), dim=1).argmax()] for sp in list(torch.unbind(solvent_pred, dim=0))]
        hd2_results_df['True_Solvent'] = list_of_true_solvent_types
        hd2_results_df.to_csv(f'predicted_binding_energies_and_designs_for_sampled_hd1_latents.csv', index=False)
    else:
        hd2_results_df = pd.read_csv(hd2_results_file)

    print(f'(DATA) Loaded hd2 results from file: {hd2_results_file} with shape {hd2_results_df.shape}')
    # Get all the points for which predicted solvent matches true solvent
    matched_solvent_df = hd2_results_df[hd2_results_df['Pred_Solvent'] == hd2_results_df['True_Solvent']]
    # Filter out only THTO solvent points
    # matched_solvent_df = matched_solvent_df[matched_solvent_df['Pred_Solvent'] == 'THTO']
    print(f'(DATA) Number of points with matched solvents: {matched_solvent_df.shape[0]}')
    dict_of_solvent_colors = {'THTO':'darkviolet', 
                                'DMSO':'royalblue', 
                                'DMF':'skyblue', 
                                'NMP':'seagreen',
                                'GBL':'gold',
                                'ACETONE':'orange',
                                'METHA':'tomato',
                                'NITRO':'indianred'}
    fig, ax = plt.subplots(figsize=(4, 3))
    c = matched_solvent_df['Pred_Solvent'].map(dict_of_solvent_colors).values
    ax.scatter(matched_solvent_df['Pred_Bandgap'], matched_solvent_df['Pred_Binding_Energy'],
                c=c, s=1, marker='.', alpha=0.7)
    ax.set_xlabel(r'Predicted Bandgap (eV)')
    ax.set_ylabel(r'Predicted Binding Energy (kJ/mol)')
    ax.set_xticks(np.arange(0, 6.0, 0.5))
    ax.set_yticks(np.arange(0, 70, 10))
    ax.set_xticklabels(np.arange(0, 6.0, 0.5).astype(str))
    ax.set_yticklabels(np.arange(0, 70, 10).astype(str))

    ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=list(dict_of_solvent_colors.keys())[i],
                    markerfacecolor=list(dict_of_solvent_colors.values())[i], markersize=6) 
                    for i in range(len(dict_of_solvent_colors))],
                    fontsize=6, ncol=1, frameon=False, bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.savefig(f'hd2_mhp_binding_energy_pareto_front.pdf', bbox_inches='tight', dpi=300)




