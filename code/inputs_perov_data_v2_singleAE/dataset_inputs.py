"""Dataset input dictionary"""

# The way this input file is structured is as a nested dictionary
# "First" level segregates dictionaries into train and predict
# "Second" level segregates different datasets under train or predict as separate dictionaries
# "Third" level allows users to select what variables become part of the dictionary

dataset_path_dict = {
    'gridSamples_200_nonlinf5': '../datasets/synthetic_dataset/synthetic_data_gridSamples_200_with_ae1_latents_concat.csv',
    'gridSamples_200_sumf5': '../datasets/synthetic_dataset/synthetic_data_gridSamples_200_sumf5_with_ae1_latents_concat.csv',
    'randomSamples_200_nonlinf5': '../datasets/synthetic_dataset/synthetic_data_randomSamples_200_with_ae1_latents_concat.csv',
    'randomSamples_200_sumf5': '../datasets/synthetic_dataset/synthetic_data_randomSamples_200_sumf5_with_ae1_latents_concat.csv',
    'PSC_bandgaps_v1': '../datasets/PSC_bandgaps/PSC_bandgaps_dataset.csv',
    'PSC_eff_v1': '../datasets/PSC_efficiencies/PSC_efficiencies_dataset.csv',
    'PSC_eff_v2': '../datasets/PSC_efficiencies/PSC_efficiencies_dataset_2.csv',
    'HSE_arun2022': '../datasets/PSC_bandgaps/HSE_data_arun2022.csv',
    'HSE_arun2024': '../datasets/PSC_bandgaps/HSE_data_arun2024.csv',
    'latents_from_SupAE1':'../runs/perovskite_multiscale_dataset_v2/SupSimpleAE_1_ldim4_arun2024/latents_from_peroveff2_film.csv',
    'latents_from_UnsupAE2':'../runs/perovskite_multiscale_dataset_v2/UnsupSimpleAE_2_ldim8_peroveff2_film/latents_from_peroveff2_film.csv',
}

list_of_nn_datasets_dict=[

        # ... Datasets to train the first autoencoder go here ...
         {

            # This will be the data used for training. Will be split into training and validation
            'train':{

                        'PSC_eff_v2':{
                              'skiprows': 1, 
                                        # desc : Number of rows to skip at the beginning of the dataset
                                        # dtype : int/None
                              'header':0, 
                                        # desc: Row number of the header
                                        # dtype : int/None
                              'path':dataset_path_dict['PSC_eff_v2'],
                                        # desc : Path to the dataset
                                        # dtype : str
                              'variables':{'all_props':{'cols':[20, 21, 22, 23, 24,
                                                                31, 32, 33, 34, 35,
                                                                42, 43, 44, 45, 46],
                                                                'preprocess': 'std'},# Preprocessing to apply to the dataset (None, std, ohe, lb, le)
                                                'perov_dep_sol':{'cols':[61], 'preprocess':'ohe'},
                                                'perov_dep_sol_mix':{'cols':[62], 'preprocess':'ohe'},
                                                'perov_dep_quench':{'cols':[67], 'preprocess':'ohe'},
                                                'phase':{'cols':[49, 50, 51, 52], 'preprocess': None},
                                                'substrate':{'cols':[78], 'preprocess':'ohe'},
                                                'etm':{'cols':[79], 'preprocess':'ohe'},
                                                'htm':{'cols':[80], 'preprocess':'ohe'},
                                                'PCE':{'cols':[92], 'preprocess':None}               
                                        },
                                'load_preprocessor':False
                                     
                                }
                        }

        }
]
