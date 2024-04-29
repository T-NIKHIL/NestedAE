"""Neural network inputs dictionary"""

list_of_nn_params_dict=[
    
       # ... NN params for the first autoencoder go here ...
       {

              # Name of the model goes here. A directory with this name will be created in the runs folder.
              'model_type':'UnsupSimpleAE_1',

              # You can create block of neural networks (called submodules) and connect them however you like.
              'submodules':{

                     'encoder':{

                            'connect_to':['X1'],
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
                                   'target':'X1'},
                            'load_params':False,
                            'save_params':False,
                            'save_output_on_fit_end':False,
                            'save_output_on_epoch_end':False
                     
                     }

              }

       },

       {

              'model_type':'SupSimpleAE_2',

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


