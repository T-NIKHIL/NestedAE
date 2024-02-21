# import tensorflow as tf
# list_of_nn_params_dict=[
        
#         {'model_name':'vanilla_ae_1',
#          'num_hidden_layers':2, 
#          'num_hidden_nodes_per_layer':[4,3], 
#          'hidden_layer_activation':['elu','elu'],
#          'latent_dim':2, 
#          'latent_layer_activation':'elu',
#          'add_supervision_on_latent':False, 
#          'y_layer_activation':None,
#          'X_hat_layer_activation':['elu'],
#          'kernel_initializer':'glorot_uniform',
#          'bias_initializer':'zeros',
#          'l1_regularization_parameter':0,
#          'l2_regularization_parameter':0.0001,         
#          'bias_regularizer':None,
#          'activity_regularizer':None},
    
#         {'model_name':'vanilla_ae_2',
#          'num_hidden_layers':1,
#          'num_hidden_nodes_per_layer':[20],
#          'hidden_layer_activation':['elu'],
#          'latent_dim':2,
#          'latent_layer_activation':'elu',
#          'add_supervision_on_latent':False,
#          'y_layer_activation':None,
#          'X_hat_layer_activation':['elu'],
#          'kernel_initializer':'glorot_uniform',
#          'bias_initializer':'zeros',
#          'l1_regularization_parameter':0.001,
#          'l2_regularization_parameter':0.001,         
#          'bias_regularizer':None,
#          'activity_regularizer':None}
# ]


# Reconstruction modules must be defined in the way inputs are fed to the model
# Any other outputs to be extracted from the module must be added before the reconstruction outputs

# Dictionary for running a nestedAE model

# If using a variational autoencoder then the submodules have fixed names
# Resampling strategy using normal dictribution : 'mu', 'logvar'

################################################################################################
# Synthetic Database dictionary
################################################################################################

list_of_nn_params_dict=[

       {
              #'model_type':'encoder_l_3_tanh_l1_1em2_decoder_None_no_l1_corr_coefs_seed_0_lr_1em2_bs_180_mae',
              'model_type':'gridSamples_nonlinf5_ae2_trial2_enc_linear_dec_linear_pred_tanh_latentd_1_bs_10_lr_0_001_w_l1_0p01_to_enc_mse',


              'submodules':{

                     'encoder':{

                            'connect_to':['x1tox8'],
                            'num_nodes_per_layer':[1],
                            'layer_type':['linear'],
                            'layer_activation':[None],
                            'layer_kernel_init':['xavier_normal'],
                            'layer_kernel_init_gain':[1],
                            'layer_bias_init':['zeros'],
                            'layer_weight_reg':{'l1':0.01, 'l2':0},
                            'save_output_on_fit_end':True,
                            'save_params':True
                     },

                     'predictor':{
                            
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[1],
                            'layer_type':['linear'],
                            'layer_activation':['tanh'],
                            'layer_kernel_init':['xavier_normal'],
                            'layer_kernel_init_gain':[1],
                            'layer_bias_init':['zeros'],
                            'layer_weight_reg':{'l1':0, 'l2':0},
                            'save_output_on_train_end':True,
                            'save_params':True,
                            'loss':{'type':'mse',
                                    'wt':1,
                                    'target':'f5'}

                     },

                     'decoder':{

                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[8],
                            'layer_type':['linear'],
                            'layer_activation':[None],
                            'layer_kernel_init':['xavier_normal'],
                            'layer_kernel_init_gain':[1],
                            'layer_bias_init':['zeros'],
                            'layer_weight_reg':{'l1':0, 'l2':0},
                            'loss':{'type':'mae',
                                    'wt':1,
                                    'target':'x1tox8'},
                            'save_params':True
                     }

              }

       },


       {
              #'model_type':'encoder_l_3_tanh_l1_1em2_decoder_None_no_l1_corr_coefs_seed_0_lr_1em2_bs_180_mae',
              #'model_type':'gridSamples_nonlinf5_ae2_trial25_enc_linear_dec_linear_pred_tanh_3n_3lay_latentd_2_bs_10_lr_0_001_w_l1_0p01_to_enc_and_to_pred',
              'model_type':'test',

              'submodules':{

                     'encoder':{

                            'connect_to':['f1tof4_w_ae1_latent'],
                            'num_nodes_per_layer':[2],
                            'layer_type':['linear'],
                            'layer_activation':[None],
                            'layer_kernel_init':['xavier_normal'],
                            'layer_kernel_init_gain':[1],
                            'layer_bias_init':['zeros'],
                            'layer_weight_reg':{'l1':0.01, 'l2':0},
                            'save_output_on_fit_end':True,
                            'save_params':True
                     },

                     'predictor':{
                         
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[3, 3, 1],
                            'layer_type':['linear', 'linear', 'linear'],
                            'layer_activation':['tanh', 'tanh', 'tanh'], 
                            'layer_kernel_init':['xavier_normal', 'xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1, 1],
                            'layer_bias_init':['zeros', 'zeros', 'zeros'],
                            'layer_weight_reg':{'l1':0.01, 'l2':0},
                            'save_output_on_fit_end':True,
                            'save_params':True,
                            'loss':{'type':'mae',
                                    'wt':1,
                                    'target':'f5'}

                     },

                     'decoder':{

                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[5],
                            'layer_type':['linear'],
                            'layer_activation':[None],
                            'layer_kernel_init':['xavier_normal'],
                            'layer_kernel_init_gain':[1],
                            'layer_bias_init':['zeros'],
                            'layer_weight_reg':{'l1':0, 'l2':0},
                            'loss':{'type':'mae',
                                    'wt':1,
                                    'target':'f1tof4_w_ae1_latent'},
                            'save_params':True
                     }

              }

       },





]