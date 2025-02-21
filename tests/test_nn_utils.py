from torch import nn, optim, zeros, equal
from torchmetrics import MeanAbsoluteError

from NestedAE.nn_utils import *
from test_inputs.sample_nn_inputs import sample_nn_params_dict
from test_inputs.sample_train_inputs import sample_nn_train_params_dict

def test_check_dict_key_exists(sample_nn_params_dict):
    assert check_dict_key_exists('modules', sample_nn_params_dict) == True

def test_get_module_input_dim_for_encoder(sample_nn_params_dict):
    module_dict_enc = sample_nn_params_dict['modules']['encoder']
    assert get_module_input_dim(module_dict_enc['connect_to'], sample_nn_params_dict, {'desc1':(100, 15)}) == 15

def test_get_module_input_dim_for_predictor(sample_nn_params_dict):
    module_dict_pred = sample_nn_params_dict['modules']['predictor']
    assert get_module_input_dim(module_dict_pred['connect_to'], sample_nn_params_dict, {'desc1':(100, 15)}) == 10

def test_set_layer_init(sample_nn_params_dict):
    enc_layer_list = [nn.Linear(15, 25),
                      nn.Tanh(),
                      nn.Linear(25, 10),
                      nn.Tanh()]
    module_dict_enc = sample_nn_params_dict['modules']['encoder']
    layer_list_init = set_layer_init(enc_layer_list, module_dict_enc)
    assert equal(layer_list_init[0].bias.data, zeros(25)) == True

def test_set_layer_activation(sample_nn_params_dict):
    activation = sample_nn_params_dict['modules']['encoder']['hidden_activation']
    assert isinstance(set_layer_activation(activation), nn.Tanh)

def test_set_layer_dropout(sample_nn_params_dict):
    dropout_type = sample_nn_params_dict['modules']['encoder']['layer_dropout']['type']
    p = sample_nn_params_dict['modules']['encoder']['layer_dropout']['p']
    assert isinstance(set_layer_dropout(dropout_type, p), nn.Dropout)

def test_create_loss_object(sample_nn_params_dict):
    loss = sample_nn_params_dict['modules']['predictor']['loss']['type']
    assert isinstance(create_loss_object(loss), nn.modules.loss.L1Loss)

def test_create_metric_object(sample_nn_params_dict):
    metric = sample_nn_params_dict['modules']['predictor']['metric'][0]
    assert isinstance(create_metric_object(metric), MeanAbsoluteError)

def test_create_optimizer_object(sample_nn_train_params_dict):
    modules = nn.ModuleDict({'encoder':nn.Linear(15, 10),
                              'predictor':nn.Linear(10, 10),
                              'decoder':nn.Linear(10, 15)})
    assert isinstance(create_optimizer_object(modules, sample_nn_train_params_dict), optim.Adam)


    







