"""Neural network inputs dictionary"""

list_of_nn_params_dict=[
    
       # ... NN params for the first autoencoder go here ...
       {

              # Name of the model goes here. A directory with this name will be created in the runs folder.
              'model_type':'singleAE_fold4',

              # You can create block of neural networks (called submodules) and connect them however you like.
              'submodules':{

                     'encoder':{
                            'connect_to':['all_props', 'perov_dep_sol', 'perov_dep_sol_mix', 'perov_dep_quench', 'phase', 'substrate', 'etm', 'htm'],
                            'num_nodes_per_layer':[50, 50, 30],
                            'layer_type':['linear', 'linear', 'linear'],
                            'layer_activation':['relu', 'relu', 'relu'],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1, 1],
                            'layer_bias_init':['zeros', 'zeros', 'zeros'],
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'save_params':True,
                            'save_output_on_epoch_end':False,
                            'save_output_on_fit_end':False
                     },

                     'pce_predictor':{
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[50, 50, 1],
                            'layer_type':['linear', 'linear', 'linear'],
                            'layer_activation':['relu', 'relu', 'relu'],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1, 1],
                            'layer_bias_init':['zeros', 'zeros', 'zeros'],
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'save_params':True,
                            'save_output_on_epoch_end':False,
                            'save_output_on_fit_end':False,
                            'loss':{'type':'mae',
                                   'wt':1,
                                   'target':'PCE'},
                     },

                     'all_props_reconst':{
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[50, 50, 15],
                            'layer_type':['linear', 'linear', 'linear'],
                            'layer_activation':['relu', 'relu', None],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1, 1],
                            'layer_bias_init':['zeros', 'zeros', 'zeros'],
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'save_params':True,
                            'save_output_on_epoch_end':False,
                            'save_output_on_fit_end':False,
                            'loss':{'type':'mae',
                                   'wt':1,
                                   'target':'all_props'},
                     },

                     'perov_dep_sol_reconst':{
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[50, 50, 4],
                            'layer_type':['linear', 'linear', 'linear'],
                            'layer_activation':['relu', 'relu', None],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1, 1],
                            'layer_bias_init':['zeros', 'zeros', 'zeros'],
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'save_params':True,
                            'save_output_on_epoch_end':False,
                            'save_output_on_fit_end':False,
                            'loss':{'type':'ce',
                                   'wt':1,
                                   'target':'perov_dep_sol'},
                     },

                     'perov_dep_sol_mix_reconst':{
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[50, 50, 10],
                            'layer_type':['linear', 'linear', 'linear'],
                            'layer_activation':['relu', 'relu', None],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1, 1],
                            'layer_bias_init':['zeros', 'zeros', 'zeros'],
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'save_params':True,
                            'save_output_on_epoch_end':False,
                            'save_output_on_fit_end':False,
                            'loss':{'type':'ce',
                                   'wt':1,
                                   'target':'perov_dep_sol_mix'},
                     },

                     'perov_dep_quench_reconst':{
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[50, 50, 6],
                            'layer_type':['linear', 'linear', 'linear'],
                            'layer_activation':['relu', 'relu', None],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1, 1],
                            'layer_bias_init':['zeros', 'zeros', 'zeros'],
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'save_params':True,
                            'save_output_on_epoch_end':False,
                            'save_output_on_fit_end':False,
                            'loss':{'type':'ce',
                                   'wt':1,
                                   'target':'perov_dep_quench'},
                     },

                     'phase_reconst':{
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[50, 50, 4],
                            'layer_type':['linear', 'linear', 'linear'],
                            'layer_activation':['relu', 'relu', None],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1, 1],
                            'layer_bias_init':['zeros', 'zeros', 'zeros'],
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'save_params':True,
                            'save_output_on_epoch_end':False,
                            'save_output_on_fit_end':False,
                            'loss':{'type':'ce',
                                   'wt':1,
                                   'target':'phase'},
                     },

                     'substrate_reconst':{
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[50, 50, 3],
                            'layer_type':['linear', 'linear', 'linear'],
                            'layer_activation':['relu', 'relu', None],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1, 1],
                            'layer_bias_init':['zeros', 'zeros', 'zeros'],
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'save_params':True,
                            'save_output_on_epoch_end':False,
                            'save_output_on_fit_end':False,
                            'loss':{'type':'ce',
                                   'wt':1,
                                   'target':'substrate'},
                     },

                     'etm_reconst':{
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[50, 50, 8],
                            'layer_type':['linear', 'linear', 'linear'],
                            'layer_activation':['relu', 'relu', None],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1, 1],
                            'layer_bias_init':['zeros', 'zeros', 'zeros'],
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'save_params':True,
                            'save_output_on_epoch_end':False,
                            'save_output_on_fit_end':False,
                            'loss':{'type':'ce',
                                   'wt':1,
                                   'target':'etm'},
                     },

                     'htm_reconst':{
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[50, 50, 4],
                            'layer_type':['linear', 'linear', 'linear'],
                            'layer_activation':['relu', 'relu', None],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1, 1],
                            'layer_bias_init':['zeros', 'zeros', 'zeros'],
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'save_params':True,
                            'save_output_on_epoch_end':False,
                            'save_output_on_fit_end':False,
                            'loss':{'type':'ce',
                                   'wt':1,
                                   'target':'htm'},
                     },

              }

       }

]

# Hyperparams optimized 
# 'encoder':{
#        'connect_to':['all_props', 'phase', 'substrate', 'etm', 'htm'],
#        'hidden_dim':{'values':[25, 50]},
#        'hidden_layers':{'values':[1, 2]},
#        'output_dim':{'values':[10, 15, 20]},
#        'layer_type':{'value':'linear'}, 
#        'layer_activation':{'values':['tanh', 'relu']},
#        'layer_kernel_init':{'value':'xavier_normal'},
#        'layer_kernel_init_gain':{'value':1},
#        'layer_bias_init':{'value':'zeros'},
#        'layer_weight_reg_l1':{'value':0},
#        'layer_weight_reg_l2':{'value':0.001}
# },


