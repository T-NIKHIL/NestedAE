"""Neural network inputs dictionary"""

list_of_nn_params_dict=[
    
       # ... NN params for the first autoencoder go here ...
       {

              # Name of the model goes here. A directory with this name will be created in the runs folder.
              'model_type':'singleAE',

              # You can create block of neural networks (called submodules) and connect them however you like.
              'submodules':{

                     'encoder':{
                            'connect_to':['all_props', 'phase', 'substrate', 'etm', 'htm'],
                            'hidden_dim':{'values':[25, 50]},
                            'hidden_layers':{'values':[1, 2]},
                            'output_dim':{'values':[10, 15, 20]},
                            'layer_type':{'value':'linear'}, 
                            'layer_activation':{'values':['tanh', 'relu']},
                            'layer_kernel_init':{'value':'xavier_normal'},
                            'layer_kernel_init_gain':{'value':1},
                            'layer_bias_init':{'value':'zeros'},
                            'layer_weight_reg_l1':{'value':0},
                            'layer_weight_reg_l2':{'value':0.001}
                     },

                     'pce_predictor':{
                            'connect_to':['encoder'],
                            'hidden_dim':'mirror_encoder',
                            'hidden_layers':'mirror_encoder',
                            'output_dim':{'value':1},
                            'output_activation':{'value':'relu'},
                            'layer_type':'mirror_encoder',
                            'layer_activation':'mirror_encoder',
                            'layer_kernel_init':'mirror_encoder',
                            'layer_kernel_init_gain':'mirror_encoder',
                            'layer_bias_init':'mirror_encoder',
                            'layer_weight_reg_l1':'mirror_encoder',
                            'layer_weight_reg_l2':'mirror_encoder',
                            'loss':{'type':'mae',
                                   'wt':1,
                                   'target':'PCE'},
                     },

                     'all_props_reconst':{
                            'connect_to':['encoder'],
                            'hidden_dim':'mirror_encoder',
                            'hidden_layers':'mirror_encoder',
                            'output_dim':{'value':15},
                            'output_activation':{'value':None},
                            'layer_type':'mirror_encoder',
                            'layer_activation':'mirror_encoder',
                            'layer_kernel_init':'mirror_encoder',
                            'layer_kernel_init_gain':'mirror_encoder',
                            'layer_bias_init':'mirror_encoder',
                            'layer_weight_reg_l1':'mirror_encoder',
                            'layer_weight_reg_l2':'mirror_encoder',
                            'loss':{'type':'mae',
                                   'wt':1,
                                   'target':'all_props'},
                     },

                     'phase_reconst':{
                            'connect_to':['encoder'],
                            'hidden_dim':'mirror_encoder',
                            'hidden_layers':'mirror_encoder',
                            'output_dim':{'value':4},
                            'output_activation':{'value':None},
                            'layer_type':'mirror_encoder',
                            'layer_activation':'mirror_encoder',
                            'layer_kernel_init':'mirror_encoder',
                            'layer_kernel_init_gain':'mirror_encoder',
                            'layer_bias_init':'mirror_encoder',
                            'layer_weight_reg_l1':'mirror_encoder',
                            'layer_weight_reg_l2':'mirror_encoder',
                            'loss':{'type':'ce',
                                   'wt':1,
                                   'target':'phase'},
                     },

                     'substrate_reconst':{
                            'connect_to':['encoder'],
                            'hidden_dim':'mirror_encoder',
                            'hidden_layers':'mirror_encoder',
                            'output_dim':{'value':3},
                            'output_activation':{'value':None},
                            'layer_type':'mirror_encoder',
                            'layer_activation':'mirror_encoder',
                            'layer_kernel_init':'mirror_encoder',
                            'layer_kernel_init_gain':'mirror_encoder',
                            'layer_bias_init':'mirror_encoder',
                            'layer_weight_reg_l1':'mirror_encoder',
                            'layer_weight_reg_l2':'mirror_encoder',
                            'loss':{'type':'ce',
                                   'wt':1,
                                   'target':'substrate'},
                     },

                     'etm_reconst':{
                            'connect_to':['encoder'],
                            'hidden_dim':'mirror_encoder',
                            'hidden_layers':'mirror_encoder',
                            'output_dim':{'value':8},
                            'output_activation':{'value':None},
                            'layer_type':'mirror_encoder',
                            'layer_activation':'mirror_encoder',
                            'layer_kernel_init':'mirror_encoder',
                            'layer_kernel_init_gain':'mirror_encoder',
                            'layer_bias_init':'mirror_encoder',
                            'layer_weight_reg_l1':'mirror_encoder',
                            'layer_weight_reg_l2':'mirror_encoder',
                            'loss':{'type':'ce',
                                   'wt':1,
                                   'target':'etm'},
                     },

                     'htm_reconst':{
                            'connect_to':['encoder'],
                            'hidden_dim':'mirror_encoder',
                            'hidden_layers':'mirror_encoder',
                            'output_dim':{'value':4},
                            'output_activation':{'value':None},
                            'layer_type':'mirror_encoder',
                            'layer_activation':'mirror_encoder',
                            'layer_kernel_init':'mirror_encoder',
                            'layer_kernel_init_gain':'mirror_encoder',
                            'layer_bias_init':'mirror_encoder',
                            'layer_weight_reg_l1':'mirror_encoder',
                            'layer_weight_reg_l2':'mirror_encoder',
                            'loss':{'type':'ce',
                                   'wt':1,
                                   'target':'htm'},
                     },

              }

       }

]


