"""Neural network inputs dictionary"""

list_of_nn_params_dict=[
    
       # ... NN params for the first autoencoder go here ...
       {

              # Name of the model goes here. A directory with this name will be created in the runs folder.
              'model_type':'UnsupSimpleAE_1_grid_lin',

              # You can create block of neural networks (called submodules) and connect them however you like.
              'submodules':{

                     'encoder':{
                            'connect_to':['X1'],
                            'hidden_dim':{'values':[25, 50]},
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

                     'decoder':{
                            'connect_to':['encoder'],
                            'hidden_dim':{'values':[25, 50]},
                            'hidden_layers':{'values':[0, 1, 2]},
                            'output_dim':{'value':8},
                            'layer_type':{'value':'linear'},
                            'layer_activation':{'values':['tanh', 'relu']},
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

       },

       {

              'model_type':'SupSimpleAE_2_grid_lin',

              'submodules':{
                  
                     'encoder':{
                            'connect_to':['X2_L1'],
                            'num_nodes_per_layer':None,
                            'layer_type':['linear'], 
                            'layer_activation':['relu'],
                            'layer_kernel_init':['xavier_normal'],
                            'layer_kernel_init_gain':[1],
                            'layer_bias_init':['zeros'],
                            'layer_weight_reg':{'l1':0, 'l2':0.001},
                            'layer_dropout':None,
                            'loss':None,
                            'load_params':False, 
                            'save_params':False,
                            'save_output_on_fit_end':False,
                            'save_output_on_epoch_end':False
                     },

                     'predictor':{
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':None,
                            'layer_type':['linear'], 
                            'layer_activation':['relu'],
                            'layer_kernel_init':['xavier_normal'],
                            'layer_kernel_init_gain':[1],
                            'layer_bias_init':['zeros'],
                            'layer_weight_reg':{'l1':0, 'l2':0.001},
                            'layer_dropout':None,
                            'loss':{'type':'mae',
                                   'wt':1,
                                   'target':'f5'},
                            'load_params':False, 
                            'save_params':False,
                            'save_output_on_fit_end':False,
                            'save_output_on_epoch_end':False
                     },

                     'decoder':{
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':None,
                            'layer_type':['linear'],
                            'layer_activation':['relu'],
                            'layer_kernel_init':['xavier_normal'],
                            'layer_kernel_init_gain':[1],
                            'layer_bias_init':['zeros'],
                            'layer_weight_reg':{'l1':0, 'l2':0.001},
                            'layer_dropout':None,
                            'loss':{'type':'mae',
                                   'wt':1,
                                   'target':'X2_L1'},
                            'load_params':False,
                            'save_params':False,
                            'save_output_on_fit_end':False,
                            'save_output_on_epoch_end':False
                     
                     }
                  
              }

       }

]


