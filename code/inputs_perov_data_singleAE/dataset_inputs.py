"""Dataset input dictionary"""

# The way this input file is structured is as a nested dictionary
# "First" level segregates dictionaries into train and predict
# "Second" level segregates different datasets under train or predict as separate dictionaries
# "Third" level allows users to select what variables become part of the dictionary

list_of_nn_datasets_dict=[
         {

            # This will be the data used for training. Will be split into training and validation
            'train':{

                        'X':{
                              'skiprows':None,
                              'header':0,
                              'path':'../datasets/PSC_efficiencies/PSC_efficiencies_dataset.csv',
                              #'weight_samples':{'col_idx':3, 'nbins':50, 'scheme':'bin_prob'},
                              'variables':{'all_props':{'cols':[20, # A_ion_rad
                                                                21, # A_at_wt
                                                                22, # A_EA
                                                                23, # A_IE
                                                                24, # A_En
                                                                31, # B_ion_rad
                                                                32, # B_at_wt
                                                                33, # B_EA
                                                                34, # B_IE
                                                                35, # B_En
                                                                42, # X_ion_rad
                                                                43, # X_at_wt
                                                                44, # X_EA
                                                                45, # X_IE
                                                                46, # X_En
                                                                ], 'preprocess':'std'},
                                          'etm':{'cols':[56], 'preprocess':'ohe'},
                                          'htm':{'cols':[57], 'preprocess':'ohe'},
                                          'PCE':{'cols':[69], 'preprocess':None}},
                              'load_preprocessor':False
                                        }

                            }
        }
]
