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

                        'X1':{
                              'skiprows': None, 
                                        # desc : Number of rows to skip at the beginning of the dataset
                                        # dtype : int/None
                              'header':0, 
                                        # desc: Row number of the header
                                        # dtype : int/None
                              'path':dataset_path_dict['gridSamples_200_sumf5'],
                                        # desc : Path to the dataset
                                        # dtype : str
                              'variables':{'X1':{'cols':[0,1,2,3,4,5,6,7], 'preprocess': 'std'}},
                              'load_preprocessor':False
                                        }

                    },
        },

        # ... Datasets to train the second autoencoder go here ...
        {

            # This will be the data used for training. Will be split into training and validation
            'train':{

                        'X2_L1':{
                              'skiprows': None, # Number of rows to skip at the beginning of the dataset
                              'header':0, # Row number of the header
                              'path':dataset_path_dict['gridSamples_200_sumf5'],
                              'variables':{'X2_L1':{'cols':[8,9,10,11,13], 'preprocess': 'std'}, 
                                           'f5':{'cols':[12], 'preprocess':'std'}},
                              'load_preprocessor':False},
                                        
                },      

        }
]
