import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import torch
from torch.nn import Module, ModuleList, ModuleDict, Linear
import umap

from NestedAE.nn_utils import check_dict_key_exists, set_layer_init

class Arctanh(torch.nn.Module):
    def forward(self, x):
        return torch.atanh(torch.clamp(x, -0.999999, 0.999999))
    
atanh_act_fn = Arctanh()

class AE(Module):
    def __init__(self, module_params):
        super(AE, self).__init__()
        ae_modules = {}
        # Outer loop iterates over the ae_modules
        for module_name, module_dict in module_params['modules'].items():
            layer_list = ModuleList()
            # Check for existence of keys or take defualts if not present
            if check_dict_key_exists('hidden_layers', module_dict):
                hidden_layers = module_dict['hidden_layers']
            else:
                hidden_layers = 0
            if check_dict_key_exists('hidden_dim', module_dict):
                hidden_dim = module_dict['hidden_dim']
            else:
                hidden_dim = None
            if check_dict_key_exists('hidden_activation', module_dict):
                hidden_activation = module_dict['hidden_activation']
            else:
                hidden_activation = None
            if check_dict_key_exists('output_activation', module_dict):
                output_activation = module_dict['output_activation']
            else:
                output_activation = None
            if check_dict_key_exists('layer_dropout', module_dict):
                layer_dropout = module_dict['layer_dropout']
            else:
                layer_dropout = None
            if check_dict_key_exists('layer_kernel_init', module_dict):
                layer_kernel_init = module_dict['layer_kernel_init']
            else:
                layer_kernel_init = None
            if check_dict_key_exists('layer_bias_init', module_dict):
                layer_bias_init = module_dict['layer_bias_init']
            else:
                layer_bias_init = None
            if check_dict_key_exists('load_params', module_dict):
                load_params = module_dict['load_params']
            else:
                load_params = False

            num_layers = hidden_layers + 1
            for layer_num in range(num_layers):
                if layer_num == 0:
                    # Calculate the input dimensions to first layer
                    input_dim = module_dict['input_dim']

                    if hidden_dim is not None:
                        layer_list.append(Linear(in_features=input_dim,
                                                out_features=module_dict['hidden_dim'],
                                                bias=True))
                    else:
                        layer_list.append(Linear(in_features=input_dim,
                                                out_features=module_dict['output_dim'],
                                                bias=True))
                        if output_activation:
                            layer_list.append(output_activation)
                        break # Only output layer
                elif layer_num == num_layers - 1:
                    layer_list.append(Linear(in_features=module_dict['hidden_dim'],
                                                out_features=module_dict['output_dim'],
                                                bias=True))
                    if output_activation:
                        layer_list.append(output_activation)
                    break # Dont add hidden activations
                else:
                    layer_list.append(Linear(in_features=module_dict['hidden_dim'],
                                                out_features=module_dict['hidden_dim'],
                                                bias=True))
                # Add hidden activations if specified
                if hidden_activation:
                    layer_list.append(hidden_activation)
                if layer_dropout:
                    layer_list.append(layer_dropout)
            # Initialize weights for all layers
            if layer_kernel_init:
                layer_list = set_layer_init(layer_list, module_dict, init='kernel')
            if layer_bias_init:
                layer_list = set_layer_init(layer_list, module_dict, init='bias')

            # Finally add to ae_module list
            ae_modules[module_name] = layer_list
        self.ae_modules = ModuleDict(ae_modules)

if __name__ == '__main__':

    # ---------------------------------------------------------------
    # --------------------- START OF USER INPUT ---------------------
    # ---------------------------------------------------------------

    random_state = 42

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

    latent_dim_for_hd2 = 6
    fold_num_for_hd2 = 7
    module_params_for_hd2 = {'name':'AE1', 
                            'modules':{
                                    
                                'encoder':{
                                    'input_dim':16,
                                    'output_dim':latent_dim_for_hd2, #8 
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

    latent_dim_for_hd1 = 10
    fold_num_for_hd1 = 2
    module_params_for_hd1 = {'name':'AE1',
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
                                    # 'layer_dropout':torch.nn.Dropout(p=0.5),
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'},

                                'A_predictor':{
                                    'input_dim':latent_dim_for_hd1,
                                    'output_dim':5,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':None,
                                    # 'layer_dropout':torch.nn.Dropout(p=0.5),
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
                                    # 'layer_dropout':torch.nn.Dropout(p=0.5),
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
                                    # 'layer_dropout':torch.nn.Dropout(p=0.5),
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'
                                }}}

    sample_hd1_latents = False
    latent_dim_of_hd1_to_sample = 0

    sample_hd2_latents = False
    latent_dim_of_hd2_to_sample = 3

    # How to sample the latents
    samples_along_selected_ldim = np.arange(-1, 1.1, 0.1).reshape(-1, 1)
    latent_samples = np.zeros((samples_along_selected_ldim.shape[0], 6))

    sample_hd2_latents_using_DOE = True
    num_levels = 8
    use_umap = True
    run_umap = True

    compare_hd1_latent_trajectories = False
    trajectory_files_for_hd1 = ['predicted_hd1_latents_from_hd2_latent_samples_along_dim1.npy',
                                'predicted_hd1_latents_from_hd2_latent_samples_along_dim4.npy']

    # -------------------------------------------------------------
    # --------------------- END OF USER INPUT ---------------------
    # -------------------------------------------------------------

    loaded_state_dict_for_hd2 = f'fold{fold_num_for_hd2}_soft_constraints_{latent_dim_for_hd2}D'
    loaded_hd2 = AE(module_params_for_hd2)
    loaded_hd2.load_state_dict(torch.load(loaded_state_dict_for_hd2))

    loaded_state_dict_for_hd1 = f'fold{fold_num_for_hd1}_soft_constraints_{latent_dim_for_hd1}D'
    loaded_hd1 = AE(module_params_for_hd1)
    loaded_hd1.load_state_dict(torch.load(loaded_state_dict_for_hd1))

    if sample_hd2_latents:
        loaded_hd2.eval()
        latent_samples[:, latent_dim_of_hd2_to_sample] = samples_along_selected_ldim[:, 0]
        # print(latent_samples)
        with torch.no_grad():
            latent_samples_tensor = torch.from_numpy(latent_samples).to(dtype=torch.float32)
            for i, layer in enumerate(loaded_hd2.ae_modules['BE_predictor']):
                if i == 0: 
                    be_pred = layer(latent_samples_tensor)
                else: 
                    be_pred = layer(be_pred)

            for i, layer in enumerate(loaded_hd2.ae_modules['latents_predictor']):
                if i == 0: 
                    hd1_latents_pred = layer(latent_samples_tensor)
                else: 
                    hd1_latents_pred = layer(hd1_latents_pred)

            for i, layer in enumerate(loaded_hd2.ae_modules['solvent_predictor']):
                if i == 0: 
                    solvent_pred = layer(latent_samples_tensor)
                else: 
                    solvent_pred = layer(solvent_pred)

    be_pred = np.round(be_pred.numpy(), 2)

    # Send the predicted HD1 latents to decoders of HD1
    loaded_hd1.eval()
    with torch.no_grad():
        for i, layer in enumerate(loaded_hd1.ae_modules['A_predictor']):
            if i == 0:
                a_pred = layer(hd1_latents_pred)
            else:
                a_pred = layer(a_pred)

        for i, layer in enumerate(loaded_hd1.ae_modules['B_predictor']):
            if i == 0:
                b_pred = layer(hd1_latents_pred)
            else:
                b_pred = layer(b_pred)

        for i, layer in enumerate(loaded_hd1.ae_modules['X_predictor']):
            if i == 0:
                x_pred = layer(hd1_latents_pred)
            else:
                x_pred = layer(x_pred)

        for i, layer in enumerate(loaded_hd1.ae_modules['bandgaps_predictor']):
            if i == 0:
                bg_pred = layer(hd1_latents_pred)
            else:
                bg_pred = layer(bg_pred)

    list_of_A_ions = ['K', 'Rb', 'Cs', 'MA', 'FA']
    list_of_B_ions = ['Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb']
    list_of_X_ions = ['Cl', 'Br', 'I']
    a_pred = np.round(torch.softmax(a_pred, dim=1).numpy(), 2)
    b_pred = np.round(torch.softmax(b_pred, dim=1).numpy(), 2)
    x_pred = np.round(torch.softmax(x_pred, dim=1).numpy(), 2)
    bg_pred = np.round(bg_pred.numpy(), 2)

    # print(list_of_A_ions)
    # print(a_pred)
    # print('\n')
    # print(list_of_B_ions)
    # print(b_pred)
    # print('\n')
    # print(list_of_X_ions)
    # print(x_pred)
    # print('\n')

    fig = plt.figure(figsize=(4, 10))
    gs = GridSpec(nrows=5, ncols=1, height_ratios=[1, 1, 1, 1, 1], hspace=0.0)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[3, 0])
    ax4 = fig.add_subplot(gs[4, 0])

    ax = [ax0, ax1, ax2, ax3, ax4]

    for a in ax:
        a.tick_params(axis='both', which='major', direction='in', length=3, width=1)
        a.tick_params(axis='both', which='minor', direction='in', length=3, width=1)
        a.xaxis.set_ticks_position('bottom')
        a.yaxis.set_ticks_position('left')

    list_of_solvent_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    list_of_solvents = ['DMSO', 'THTO', 'DMF', 'NMP', 'ACETONE', 'METHA', 'GBL', 'NITRO']
    solvent_pred_decoded = [list_of_solvents[i] for i in list(torch.softmax(solvent_pred, dim=1).argmax(dim=1))]
    print('\n')
    plot_solvent_colors = []
    for solvent in solvent_pred_decoded:
        plot_solvent_colors.append(list_of_solvent_colors[list_of_solvents.index(solvent)])

    ax0.scatter(samples_along_selected_ldim, be_pred, color=plot_solvent_colors, marker='.')
    ax0.set_ylabel('Predicted Binding Energy (eV)')
    ax0.set_xticks(np.arange(-1, 1.2, 0.2))
    ax0.set_xticklabels([])
    ax0.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=list_of_solvents[i],
                        markerfacecolor=list_of_solvent_colors[i], markersize=6) 
                        for i in range(len(list_of_solvents))],
                        fontsize=6, ncol=4, frameon=False, loc='upper right', bbox_to_anchor=(1, 1))

    ax1.scatter(samples_along_selected_ldim, bg_pred, color='k', marker='.')
    ax1.set_ylabel('Predicted Bandgap (eV)')
    ax1.set_xticks(np.arange(-1, 1.2, 0.2))
    ax1.set_xticklabels([])

    for i in range(a_pred.shape[1]):
        ax2.scatter(samples_along_selected_ldim, a_pred[:, i], marker='.', label=list_of_A_ions[i])
    ax2.set_ylabel('Predicted A-site conc.')
    ax2.set_yticks(np.arange(0, 1.0, 0.2))
    ax2.set_yticklabels([f'{y:.1f}' for y in np.arange(0, 1.0, 0.2)])
    ax2.set_ylim(bottom=-0.1, top=1)
    ax2.set_xticks(np.arange(-1, 1.2, 0.2))
    ax2.set_xticklabels([])
    ax2.legend(fontsize=6, ncol=5, frameon=False)

    for i in range(b_pred.shape[1]):
        ax3.scatter(samples_along_selected_ldim, b_pred[:, i], marker='.', label=list_of_B_ions[i]) 
    ax3.set_ylabel('Predicted B-site conc.')
    ax3.set_yticks(np.arange(0, 1.0, 0.2))
    ax3.set_yticklabels([f'{y:.1f}' for y in np.arange(0, 1.0, 0.2)])
    ax3.set_ylim(bottom=-0.1, top=1)
    ax3.set_xticks(np.arange(-1, 1.2, 0.2))
    ax3.set_xticklabels([])
    ax3.legend(fontsize=6, ncol=6, frameon=False)

    for i in range(x_pred.shape[1]):
        ax4.scatter(samples_along_selected_ldim, x_pred[:, i], marker='.', label=list_of_X_ions[i])
    ax4.set_ylabel('Predicted X-site conc.')
    ax4.set_yticks(np.arange(0, 1.0, 0.2))
    ax4.set_yticklabels([f'{y:.1f}' for y in np.arange(0, 1.0, 0.2)])
    ax4.set_ylim(bottom=-0.1, top=1)
    ax4.set_xlabel(f'Latent dimension {latent_dim_of_hd2_to_sample + 1} values')
    ax4.legend(fontsize=6, ncol=3, frameon=False)
    ax4.set_xticks(np.arange(-1, 1.2, 0.2))
    ax4.set_xticklabels([f'{x:.1f}' for x in np.arange(-1, 1.2, 0.2)])
    # ax4.set_xlim([-1, 1])
    plt.tight_layout()
    plt.savefig(f'hd1_and_hd2_design_and_prop_preds_from_hd2_latent_samples_along_dim{latent_dim_of_hd2_to_sample + 1}.pdf', bbox_inches='tight', dpi=300)

    np.save(f'predicted_hd1_latents_from_hd2_latent_samples_along_dim{latent_dim_of_hd2_to_sample + 1}.npy', hd1_latents_pred.numpy())

    if sample_hd2_latents_using_DOE:
        num_experiments = num_levels**latent_dim_for_hd2
        print(f'Number of experiments to run using DOE with {num_levels} levels & {latent_dim_for_hd2} features: {num_experiments}')
        samples = np.arange(-1, 1.01, 2/(num_levels-1))
        latent_samples = np.zeros((num_experiments, latent_dim_for_hd2))
        for i in range(latent_dim_for_hd2):
            # print(np.repeat(samples, num_levels**i)[0:100])
            latent_samples[:, i] = np.tile(np.repeat(samples, num_levels**i), int(num_experiments/(num_levels**(i+1))))

        loaded_hd2.eval()
        with torch.no_grad():
            latent_samples_tensor = torch.from_numpy(latent_samples).to(dtype=torch.float32)
            for i, layer in enumerate(loaded_hd2.ae_modules['BE_predictor']):
                if i == 0: 
                    be_pred = layer(latent_samples_tensor)
                else: 
                    be_pred = layer(be_pred)

            for i, layer in enumerate(loaded_hd2.ae_modules['latents_predictor']):
                if i == 0: 
                    hd1_latents_pred = layer(latent_samples_tensor)
                else: 
                    hd1_latents_pred = layer(hd1_latents_pred)

            for i, layer in enumerate(loaded_hd2.ae_modules['solvent_predictor']):
                if i == 0: 
                    solvent_pred = layer(latent_samples_tensor)
                else: 
                    solvent_pred = layer(solvent_pred)
    
        be_pred = np.round(be_pred.numpy(), 2)

        list_of_solvent_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
        list_of_solvents = ['DMSO', 'THTO', 'DMF', 'NMP', 'ACETONE', 'METHA', 'GBL', 'NITRO']
        solvent_pred_decoded = [list_of_solvents[i] for i in list(torch.softmax(solvent_pred, dim=1).argmax(dim=1))]
        plot_solvent_colors = []
        for solvent in solvent_pred_decoded:
            plot_solvent_colors.append(list_of_solvent_colors[list_of_solvents.index(solvent)])

        if use_umap:
            if run_umap:
                reducer = umap.UMAP(n_neighbors=20, n_components=2, metric='', n_epochs=200, learning_rate=1.0, min_dist=0.1, spread=1.0, random_state=random_state)
                embedded_dataset = reducer.fit_transform(latent_samples)
                embedded_dataset_df = pd.DataFrame(np.concatenate((embedded_dataset, be_pred), axis=1), columns=['UMAP1', 'UMAP2', 'Predicted Binding Energy'])
                embedded_dataset_df.to_csv(f'umap_coords_using_{num_levels}_levels.csv', index=False)
                dim1 = embedded_dataset_df['UMAP1'].values
                dim2 = embedded_dataset_df['UMAP2'].values
            else:
                loaded_embedded_dataset_df = pd.read_csv(f'umap_coords_using_{num_levels}_levels.csv')
                dim1 = loaded_embedded_dataset_df['UMAP1'].values
                dim2 = loaded_embedded_dataset_df['UMAP2'].values

        # Scatter plot true and predicted bandgaps
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        sc = ax[0].scatter(dim1, dim2, alpha=1.0, s=2, c=be_pred, cmap='viridis')
        ax[0].set_xlabel(r'UMAP Dimension 1')
        ax[0].set_ylabel(r'UMAP Dimension 2')
        ax[0].set_xticks(np.arange(dim1.min(), dim1.max()+0.1, 0.5))
        ax[0].set_yticks(np.arange(dim2.min(), dim2.max()+0.1, 0.5))
        ax[0].set_xticklabels([f'{x:.1f}' for x in np.arange(dim1.min(), dim1.max()+0.1, 0.5)])
        ax[0].set_yticklabels([f'{y:.1f}' for y in np.arange(dim2.min(), dim2.max()+0.1, 0.5)])
        cbar = plt.colorbar(sc, ax=ax[0])
        cbar.set_label(r'Predicted Binding Energy (kJ/mol)', rotation=270, labelpad=15)
        sc2 = ax[1].scatter(dim1, dim2, alpha=1.0, s=2, c=plot_solvent_colors)
        ax[1].set_xlabel(r'UMAP Dimension 1')
        ax[1].set_ylabel(r'UMAP Dimension 2')
        # cbar2 = plt.colorbar(sc2, ax=ax[1], ticks=np.arange(len(list_of_solvents)))
        # cbar2.ax.set_yticklabels(list_of_solvents)
        # cbar2.set_label(r'Solvent', rotation=270, labelpad=15)
        # Create a colorbar with discrete colors representing each solvent
        ax[1].legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=list_of_solvents[i],
                            markerfacecolor=list_of_solvent_colors[i], markersize=6) 
                            for i in range(len(list_of_solvents))],
                            fontsize=6, ncol=1, frameon=False, bbox_to_anchor=(1.01, 1))
        ax[1].set_xticks(np.arange(dim1.min(), dim1.max()+0.1, 0.5))
        ax[1].set_yticks(np.arange(dim2.min(), dim2.max()+0.1, 0.5))
        ax[1].set_xticklabels([f'{x:.1f}' for x in np.arange(dim1.min(), dim1.max()+0.1, 0.5)])
        ax[1].set_yticklabels([f'{y:.1f}' for y in np.arange(dim2.min(), dim2.max()+0.1, 0.5)])
        ax[1].set_aspect('equal', 'box')
        ax[0].set_aspect('equal', 'box')

        plt.tight_layout()
        plt.savefig(f'hd2_sampled_latents_using_{num_levels}_level_doe_umap_embeddings.pdf', bbox_inches='tight', dpi=300)

    if compare_hd1_latent_trajectories:
        pass

