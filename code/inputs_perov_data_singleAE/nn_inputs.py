"""Neural network inputs dictionary"""

list_of_nn_params_dict=[
       {

              'model_type':'singleAE_fold0_ldim2',

              'submodules':{

                     'encoder':{
                            'connect_to':['all_props', 'etm', 'htm'],
                            'num_nodes_per_layer':[50, 2],
                            'layer_type':['linear', 'linear'],
                            'layer_activation':['relu', 'relu'],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1],
                            'layer_bias_init':['zeros', 'zeros'],
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'save_output_on_fit_end':True,
                            'save_params':True  
                     },

                     'PCE_pred':{
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[50, 1],
                            'layer_type':['linear', 'linear'],
                            'layer_activation':['relu', 'relu'],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1],
                            'layer_bias_init':['zeros', 'zeros'],
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'save_output_on_fit_end':True,
                            'save_params':True,
                            'loss':{'type':'mae',
                                   'wt':1,
                                   'target':'PCE'},
                     },

                     'all_props_pred':{
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[50, 15],
                            'layer_type':['linear', 'linear'],
                            'layer_activation':['relu', None],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1],
                            'layer_bias_init':['zeros', 'zeros'],
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'save_output_on_fit_end':True,
                            'save_params':True,          
                            'loss':{'type':'mae',
                                    'wt':1,
                                    'target':'all_props'}
                     },
                     
                     'etm_pred':{
                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[50, 7],
                            'layer_type':['linear', 'linear'],
                            'layer_activation':['relu', None],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1],
                            'layer_bias_init':['zeros', 'zeros'],
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'save_output_on_fit_end':True,
                            'save_params':True,
                            'loss':{'type':'ce',
                                    'wt':1,
                                    'target':'etm'}
                     },

                     'htm_pred':{

                            'connect_to':['encoder'],
                            'num_nodes_per_layer':[50, 4],
                            'layer_type':['linear', 'linear'],
                            'layer_activation':['relu', None],
                            'layer_kernel_init':['xavier_normal', 'xavier_normal'],
                            'layer_kernel_init_gain':[1, 1],
                            'layer_bias_init':['zeros', 'zeros'],
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'save_output_on_fit_end':True,
                            'save_params':True,
                            'loss':{'type':'ce',
                                    'wt':1,
                                    'target':'htm'}
                     }
              }
       }
]

# input dictionary for hyperparameter tuning
# 'encoder':{

#                             'connect_to':['all_props', 'etm', 'htm'],
#                             'hidden_dim':{'values':[25, 50]},
#                             'hidden_layers':{'values':[1, 2]},
#                             'output_dim':{'values':[12, 14, 16, 18, 20, 22]},
#                             'layer_type':{'value':'linear'}, 
#                             'layer_activation':{'value':'relu'},
#                             'layer_kernel_init':{'value':'xavier_normal'},
#                             'layer_kernel_init_gain':{'value':1},
#                             'layer_bias_init':{'value':'zeros'},
#                             'layer_weight_reg_l1':{'value':0},
#                             'layer_weight_reg_l2':{'value':0.001}
#                      },

#                      'PCE_pred':{
#                            'connect_to':['encoder'],
#                            'output_dim':{'value':1},
#                            'output_activation':{'value':'relu'},
#                            'num_nodes_per_layer':'same_as_encoder',
#                            'layer_type':'same_as_encoder',
#                            'layer_activation':'same_as_encoder',
#                            'layer_kernel_init':'same_as_encoder',
#                            'layer_kernel_init_gain':'same_as_encoder',
#                            'layer_bias_init':'same_as_encoder',
#                            'layer_weight_reg_l1':'same_as_encoder',
#                            'layer_weight_reg_l2':'same_as_encoder',
#                            'loss':{'type':'mae',
#                                    'wt':1,
#                                    'target':'PCE'},
#                      },

#                      'all_props_pred':{
#                            'connect_to':['encoder'],
#                            'output_dim':{'value':15},
#                            'output_activation':{'value':None},
#                            'num_nodes_per_layer':'same_as_encoder',
#                            'layer_type':'same_as_encoder',
#                            'layer_activation':'same_as_encoder',
#                            'layer_kernel_init':'same_as_encoder',
#                            'layer_kernel_init_gain':'same_as_encoder',
#                            'layer_bias_init':'same_as_encoder',
#                            'layer_weight_reg_l1':'same_as_encoder',
#                            'layer_weight_reg_l2':'same_as_encoder',
#                            'loss':{'type':'mae',
#                                    'wt':1,
#                                    'target':'all_props'}
#                      },
                     
#                      'etm_pred':{
#                             'connect_to':['encoder'],
#                             'output_dim':{'value':7},
#                             'output_activation':{'value':None},
#                             'num_nodes_per_layer':'same_as_encoder',
#                             'layer_type':'same_as_encoder',
#                             'layer_activation':'same_as_encoder',
#                             'layer_kernel_init':'same_as_encoder',
#                             'layer_kernel_init_gain':'same_as_encoder',
#                             'layer_bias_init':'same_as_encoder',
#                             'layer_weight_reg_l1':'same_as_encoder',
#                             'layer_weight_reg_l2':'same_as_encoder',
#                             'loss':{'type':'ce',
#                                     'wt':1,
#                                     'target':'etm'}
#                      },

#                      'htm_pred':{

#                             'connect_to':['encoder'],
#                             'output_dim':{'value':4},
#                             'output_activation':{'value':None},
#                             'num_nodes_per_layer':'same_as_encoder',
#                             'layer_type':'same_as_encoder',
#                             'layer_activation':'same_as_encoder',
#                             'layer_kernel_init':'same_as_encoder',
#                             'layer_kernel_init_gain':'same_as_encoder',
#                             'layer_bias_init':'same_as_encoder',
#                             'layer_weight_reg_l1':'same_as_encoder',
#                             'layer_weight_reg_l2':'same_as_encoder',
#                             'loss':{'type':'ce',
#                                     'wt':1,
#                                     'target':'htm'}
#                      }


