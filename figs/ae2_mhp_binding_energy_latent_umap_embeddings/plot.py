import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import umap

from model import AE, Arctanh

atanh_act_fn = Arctanh()

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

    model_dir = '../../runs/ae2_bandgaps_THEN_perov_solv_BE/fold3_soft_constraints_7D'
    latent_dim = 7
    fold_num = 3
    module_params = {'name':'AE1',
                            'modules':{
                                
                                'encoder':{
                                    'input_dim':17,
                                    'output_dim':latent_dim, 
                                    'hidden_dim':25,
                                    'hidden_layers':1, 
                                    'hidden_activation':None, 
                                    'output_activation':torch.nn.Tanh(), 
                                    'layer_kernel_init':'xavier_normal', 
                                    'layer_bias_init':'zeros'},


                                'BE_predictor':{
                                    'input_dim':latent_dim,
                                    'output_dim':1,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':torch.nn.ReLU(),
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'},

                                'latents_predictor':{
                                    'input_dim':latent_dim,
                                    'output_dim':11,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':atanh_act_fn,
                                    'output_activation':None,
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'},
                                
                                'decoder':{
                                    'input_dim':latent_dim,
                                    'output_dim':6,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':None,
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'},

                                'solvent_predictor':{
                                    'input_dim':latent_dim,
                                    'output_dim':8,
                                    'hidden_dim':25,
                                    'hidden_layers':1,
                                    'hidden_activation':torch.nn.ReLU(),
                                    'output_activation':None,
                                    'layer_kernel_init':'xavier_normal',
                                    'layer_bias_init':'zeros'},
                        }
                    }

    #################################################################
    # Dataset Parameters
    #################################################################

    dataset_loc = '../../datasets/MHP_bandgaps_AND_perov_solvent_BE/nestedae_dataset/perov_solv_BE_for_nestedae.csv'
    descriptors = ['SOLV_DENSITY',
                    'SOLV_DIELECTRIC',
                    'SOLV_GDN',
                    'SOLV_DPM',
                    'SOLV_MV',
                    'SOLV_UMBO',
                    'l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10']
    latent_descriptors = ['l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10']
    target = ['Target']
    standardize_descs = True

    #################################################################
    # Script Parameters
    #################################################################

    standardize_latents_before_umap = False
    n_components = 2
    n_neighbors = 50 # Size of local neighborhood used for manifold approximation
    metric = 'euclidean' 
    n_epochs = 1000 # 200 for large and 500 for small datasets 
    init = 'spectral' # How to initialize the low-D embedding. 'spectral' or 'random'
    min_dist = 0.3 # Minimum distance between two points in the low dim space [0.1, 0.99]
    spread = 1.0 # Effective scale of embedded points
    learning_rate = 0.1 # initial learning rate for embedding algorithm
    load_data = False
    data_filename = 'embedded_data.csv'
    use_true_BE_for_coloring = True

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

        x_torch_all = torch.from_numpy(x_dataframe.to_numpy()).to(dtype=torch.float32)
        true_prop = y_dataframe.values.reshape(-1, 1)

        # Load the nested autoencoder model
        loaded_model = AE(module_params)
        loaded_model.load_state_dict(torch.load(model_dir))

        loaded_model.eval()
        with torch.no_grad():
            for i , layer in enumerate(loaded_model.ae_modules['encoder']):
                if i == 0:
                    latents = layer(x_torch_all)
                else:
                    latents = layer(latents)

            for i, layer in enumerate(loaded_model.ae_modules['BE_predictor']):
                if i == 0:
                    y1_pred = layer(latents)
                else:
                    y1_pred = layer(y1_pred)

        # Standardize the latents before UMAP
        if standardize_latents_before_umap:
            latents_stand = (latents - latents.mean(dim=0)) / latents.std(dim=0)
        else:
            latents_stand = latents

        reducer = umap.UMAP(n_components=n_components, 
                            n_neighbors=n_neighbors,
                            metric=metric, 
                            n_epochs=n_epochs,
                            init=init,
                            min_dist=min_dist, 
                            spread=spread,
                            learning_rate=learning_rate, 
                            random_state=random_state)
        embedded_dataset = reducer.fit_transform(latents_stand)

        embedded_dataset_df = pd.DataFrame(np.concatenate((embedded_dataset, 
                                                           y1_pred.numpy(), 
                                                           true_prop), axis=1), 
                                                           columns=['UMAP1', 'UMAP2', 'Predicted Binding Energy', 'True Binding Energy'])
        # Save the UMAP Embeddings to csv
        embedded_dataset_df.to_csv('embedded_data.csv', index=False)

    else:
        # Load the data from csv file
        embedded_dataset_df = pd.read_csv('embedded_data.csv')

    dim1 = embedded_dataset_df['UMAP1'].values
    dim2 = embedded_dataset_df['UMAP2'].values
    true_be = embedded_dataset_df['True Binding Energy'].values
    pred_be = embedded_dataset_df['Predicted Binding Energy'].values

    # Scatter plot true and predicted bandgaps
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    if use_true_BE_for_coloring:
        sc = ax.scatter(dim1, dim2, alpha=1.0, s=1, c=true_be, cmap='viridis')
    else:
        sc = ax.scatter(dim1, dim2, alpha=1.0, s=1, c=pred_be, cmap='viridis')
    ax.set_xlabel(r'UMAP Dimension 1')
    ax.set_ylabel(r'UMAP Dimension 2')
    cbar = plt.colorbar(sc, ax=ax)
    if use_true_BE_for_coloring:
        cbar.set_label(r'True Binding Energy (kJ/mol)', rotation=270, labelpad=15)
    else:
        cbar.set_label(r'Predicted Binding Energy (kJ/mol)', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.savefig('ae2_mhp_binding_energy_latent_umap_embeddings.pdf', bbox_inches='tight', dpi=300)