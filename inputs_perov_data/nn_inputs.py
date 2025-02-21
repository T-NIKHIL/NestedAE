"""Neural network inputs dictionary"""

# Hyperparam tuning of autoencoders done by mirroring the encoder architecture for the decoders and predictors.
# For any submodule other than encoder and if making a prediction must specify the 'output_dim', 'output_activation', and 'loss' keys.

list_of_nn_params_dict=[
    
       # ... NN params for the first autoencoder go here ...
       {
              'name':'ae1',
              'modules':{

                     'encoder':{
                            'connect_to':['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sb', 'Pb', 'Cl', 'Br', 'I'], # List of features to connect to in database go here
                            'output_dim':{'values':[8, 10, 12, 14]},
                            'hidden_dim':{'values':[25, 50, 75]}, # Use the 'values' key for hyperparam tuning
                            # 'hidden_layers':{'values':[1, 2, 3]},
                            'hidden_layers':1, # testing purposes
                            # 'hidden_activation':{'values':['tanh', 'relu']},
                            'hidden_activation':'tanh', 
                            'layer_type':'linear', 
                            'layer_kernel_init':'xavier_normal',
                            'layer_bias_init':'zeros',
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'param_optimization':False,
                            'save_output_on_fit_end':True,

                     },

                     'predictor':{
                            'connect_to':['encoder'],
                            'output_dim':1,
                            'output_activation':'relu',
                            # 'hidden_dim':'mirror', # Use 'mirror' to mirror the encoder architecture
                            'hidden_dim':75,
                            # 'hidden_layers':'mirror', # Use 'mirror' to mirror the encoder architecture
                            'hidden_layers':1, 
                            # 'hidden_activation':'mirror', # Use 'mirror' to mirror the encoder architecture
                            'hidden_activation':'tanh',
                            'layer_type':'linear',
                            'layer_kernel_init':'xavier_normal',
                            'layer_bias_init':'zeros',
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'loss':{'type':'mae',
                                   'wt':1,
                                   'target':'bg'},
                            'param_optimization':False,
                            'save_output_on_fit_end':True
                     },

                     'decoder':{
                            'connect_to':['encoder'],
                            'output_dim':15,
                            # 'hidden_dim':'mirror', # Use 'mirror' to mirror the encoder architecture
                            'hidden_dim':75,
                            # 'hidden_layers':'mirror', # Use 'mirror' to mirror the encoder architecture
                            'hidden_layers':1, 
                            # 'hidden_activation':'mirror', # Use 'mirror' to mirror the encoder architecture
                            'hidden_activation':'tanh',
                            'layer_type':'linear',
                            'layer_kernel_init':'xavier_normal',
                            'layer_bias_init':'zeros',
                            'layer_weight_reg_l1':0,
                            'layer_weight_reg_l2':0.001,
                            'loss':{'type':'mae',
                                   'wt':1,
                                   'target':'all_props'},
                            'param_optimization':False,
                            'save_output_on_fit_end':True
                     }

              }

       }

]