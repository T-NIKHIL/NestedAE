import torch
from torch.nn import Module, ModuleDict, ModuleList, Linear

from NestedAE.nn_utils import check_dict_key_exists, set_layer_init

class Arctanh(torch.nn.Module):
    def forward(self, x):
        return torch.atanh(torch.clamp(x, -0.999999, 0.999999))
    
atanh_act_fn = Arctanh()

class AE(Module):
    def __init__(self, module_params):
        super(AE, self).__init__()
        ae_modules = {}
        # Outer loop iterates over the ae_modules
        for module_name, module_dict in module_params['modules'].items():
            layer_list = ModuleList()
            # Check for existence of keys or take defualts if not present
            if check_dict_key_exists('hidden_layers', module_dict):
                hidden_layers = module_dict['hidden_layers']
            else:
                hidden_layers = 0
            if check_dict_key_exists('hidden_dim', module_dict):
                hidden_dim = module_dict['hidden_dim']
            else:
                hidden_dim = None
            if check_dict_key_exists('hidden_activation', module_dict):
                hidden_activation = module_dict['hidden_activation']
            else:
                hidden_activation = None
            if check_dict_key_exists('output_activation', module_dict):
                output_activation = module_dict['output_activation']
            else:
                output_activation = None
            if check_dict_key_exists('layer_dropout', module_dict):
                layer_dropout = module_dict['layer_dropout']
            else:
                layer_dropout = None
            if check_dict_key_exists('layer_kernel_init', module_dict):
                layer_kernel_init = module_dict['layer_kernel_init']
            else:
                layer_kernel_init = None
            if check_dict_key_exists('layer_bias_init', module_dict):
                layer_bias_init = module_dict['layer_bias_init']
            else:
                layer_bias_init = None
            if check_dict_key_exists('load_params', module_dict):
                load_params = module_dict['load_params']
            else:
                load_params = False

            num_layers = hidden_layers + 1
            for layer_num in range(num_layers):
                if layer_num == 0:
                    # Calculate the input dimensions to first layer
                    input_dim = module_dict['input_dim']

                    if hidden_dim is not None:
                        layer_list.append(Linear(in_features=input_dim,
                                                out_features=module_dict['hidden_dim'],
                                                bias=True))
                    else:
                        layer_list.append(Linear(in_features=input_dim,
                                                out_features=module_dict['output_dim'],
                                                bias=True))
                        if output_activation:
                            layer_list.append(output_activation)
                        break # Only output layer
                elif layer_num == num_layers - 1:
                    layer_list.append(Linear(in_features=module_dict['hidden_dim'],
                                                out_features=module_dict['output_dim'],
                                                bias=True))
                    if output_activation:
                        layer_list.append(output_activation)
                    break # Dont add hidden activations
                else:
                    layer_list.append(Linear(in_features=module_dict['hidden_dim'],
                                                out_features=module_dict['hidden_dim'],
                                                bias=True))
                # Add hidden activations if specified
                if hidden_activation:
                    layer_list.append(hidden_activation)
                if layer_dropout:
                    layer_list.append(layer_dropout)
            # Initialize weights for all layers
            if layer_kernel_init:
                layer_list = set_layer_init(layer_list, module_dict, init='kernel')
            if layer_bias_init:
                layer_list = set_layer_init(layer_list, module_dict, init='bias')

            # Finally add to ae_module list
            ae_modules[module_name] = layer_list
        self.ae_modules = ModuleDict(ae_modules)

if __name__ == "__main__":
    pass