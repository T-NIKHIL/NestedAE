"""Neural network inputs dictionary"""

# Hyperparam tuning of autoencoders done by mirroring the encoder architecture for the decoders and predictors.
# For any submodule other than encoder and if making a prediction must specify the 'output_dim', 'output_activation', and 'loss' keys.

list_of_nn_params_dict=[
    
       # ... NN params for the first autoencoder go here ...
       {

              # Name of the model goes here. A directory with this name will be created in the runs folder.
              'model_type':'model_name_goes_here',

              # You can create block of neural networks (called submodules) and connect them however you like.
              'submodules':{

                     'encoder':{
                            'connect_to':['feature_1', 'feature_2', ...], # List of features to connect to in database go here
                            'hidden_dim':{'values':[25, 50]}, # Use the 'values' key for hyperparam tuning otherise use 'value'
                            'hidden_layers':{'values':[0, 1, 2]},
                            'output_dim':{'values':[2, 4, 6]},
                            'layer_type':{'value':'linear'}, 
                            'layer_activation':{'values':['tanh', 'relu']},
                            'layer_kernel_init':{'value':'xavier_normal'},
                            'layer_kernel_init_gain':{'value':1},
                            'layer_bias_init':{'value':'zeros'},
                            'layer_weight_reg_l1':{'value':0},
                            'layer_weight_reg_l2':{'value':0.001}
                     },

                     'predictor':{
                            'connect_to':['encoder'],
                            'hidden_dim':'mirror', # Use 'mirror' to mirror the encoder architecture
                            'hidden_layers':'mirror',
                            'output_dim':{'value':8},
                            'output_activation':{'value':None},
                            'layer_type':{'value':'linear'},
                            'layer_activation':'mirror',
                            'layer_kernel_init':{'value':'xavier_normal'},
                            'layer_kernel_init_gain':{'value':1},
                            'layer_bias_init':{'value':'zeros'},
                            'layer_weight_reg_l1':{'value':0},
                            'layer_weight_reg_l2':{'value':0.001},
                            'loss':{'type':'mae',
                                   'wt':1,
                                   'target':'X1'},
                     },

                     'decoder':{
                            'connect_to':['encoder'],
                            'hidden_dim':'mirror', # Use 'mirror' to mirror the encoder architecture
                            'hidden_layers':'mirror',
                            'output_dim':{'value':8},
                            'output_activation':{'value':None},
                            'layer_type':{'value':'linear'},
                            'layer_activation':'mirror',
                            'layer_kernel_init':{'value':'xavier_normal'},
                            'layer_kernel_init_gain':{'value':1},
                            'layer_bias_init':{'value':'zeros'},
                            'layer_weight_reg_l1':{'value':0},
                            'layer_weight_reg_l2':{'value':0.001},
                            'loss':{'type':'mae',
                                   'wt':1,
                                   'target':'X1'},
                     }

              }

       }

]

# list_of_nn_params_dict=[
    
#        # ... NN params for the first autoencoder go here ...
#        {

#               # Name of the model goes here. A directory with this name will be created in the runs folder.
#               'model_type':'UnsupSimpleAE_1_grid_lin',

#               # You can create block of neural networks (called submodules) and connect them however you like.
#               'submodules':{

#                      'encoder':{
#                             'connect_to':['X1'],
#                             'hidden_dim':{'values':[25, 50]},
#                             'hidden_layers':{'values':[0, 1, 2]},
#                             'output_dim':{'values':[2, 4, 6]},
#                             'layer_type':{'value':'linear'}, 
#                             'layer_activation':{'values':['tanh', 'relu']},
#                             'layer_kernel_init':{'value':'xavier_normal'},
#                             'layer_kernel_init_gain':{'value':1},
#                             'layer_bias_init':{'value':'zeros'},
#                             'layer_weight_reg_l1':{'value':0},
#                             'layer_weight_reg_l2':{'value':0.001}
#                      },

#                      'decoder':{
#                             'connect_to':['encoder'],
#                             'output_dim':{'value':8},
#                             'output_activation':{'value':None},
#                             'loss':{'type':'mae',
#                                    'wt':1,
#                                    'target':'X1'},
#                      }

#               }

#        }

# ]


