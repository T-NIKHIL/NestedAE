################################################################################################
# Synthetic Database dictionary
################################################################################################

#'variables':{'f1tof4':{'cols':[0,1,2,3], 'preprocess':'std'}
list_of_nn_datasets_dict=[

    {

        'train':{
            
                    'X1':{
                            'skiprows':1, # Skips the header row
                            'header':None,
                            'path':'../datasets/synthetic_dataset/synthetic_data_gridSamples_200_sumf5.csv',
                            'variables':{'f1tof4':{'cols':[8,9,10,11], 'preprocess':'std'},
                                         'f5':{'cols':[12], 'preprocess':'std'}}

                        },


    }
    }

    # {

    #     'train':{

    #                 'X2':{
    #                         'skiprows':1,
    #                         'header':None,
    #                         'path':'../runs/noisyF_l2-5_lr_1em4_l1_1em3_l2_0_tanh_200pts/synthetic_data_FuncAddNoise_200.csv',
    #                         'variables':{'f5tof13andae1l':{'cols':[9,10,11,12,13,14,15,16,17,18,19], 'preprocess':'std'}}
    #                     }

    #     }
    # }
]
