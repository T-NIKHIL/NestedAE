" nn utilities script "

import math

from torch import nn, optim, zeros, matmul, Tensor, tensor, mean, sum
from torch import float32 as torch_float32
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, RichModelSummary # type: ignore
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme # type: ignore
from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError # type: ignore

# Binary Linear layer implementation
class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(Tensor(out_features, in_features).uniform_(-1, 1))
        if bias:
            self.bias = nn.Parameter(zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, X):

        if self.bias is not None:
            out = matmul(X, self.weight.clamp(min=0).sign().t()) + self.bias
        else:
            out = matmul(X, self.binary_weight.t())

        return out

def check_dict_key_exists(key, dictionary):
    """Check if key exists in dictionary

    Args:
        key (str) : key to check for
        dictionary (dict) : dictionary to check

    Returns: True if key exists in dictionary, False otherwise
    """
    if (key in list(dictionary.keys())) and \
        (dictionary[key] is not None):
        return True
    return False

def get_module_input_dim(connect_to, nn_params_dict, desc_shapes):
    """Get input dimension of a module.
    
    Args:
        connect_to (list) : list of modules/descriptors to connect to
        nn_params_dict (dict) : dictionary containing neural network parameters
        desc_shapes (dict) : dictionary containing shapes of descriptors
    
    Returns : tot_input_dim (int) : total input dimension of data to module
    """
    module_dicts = nn_params_dict['modules']
    tot_input_dim = 0
    for inp in connect_to:
        # Case where input to layer is the training data
        if inp in list(desc_shapes.keys()):
            input_dim = desc_shapes[inp][1]
        # Case where input to layer is ouput from last layer of connected module
        else:
            input_dim = module_dicts[inp]['output_dim']
        tot_input_dim += input_dim
    return tot_input_dim 

def set_layer_init(layer_list, module_dict):
    """Initialize layer weights and biases.
    
    Args:
        layer_list (list) : list of torch.nn.Linear layers
        module_dict (dict) : dictionary containing module parameters

    Returns :
        layer_list (list) : list of torch.nn.Linear layers with initialized weights and biases
    """
    layers = [layer for layer in layer_list \
                if isinstance(layer, nn.modules.linear.Linear)]
    # Calculating gain for xavier init of hidden layers
    if module_dict['hidden_activation'] == 'relu':
        hidden_gain = math.sqrt(2)
    elif module_dict['hidden_activation'] == 'tanh':
        hidden_gain = 5.0 / 3
    else:
        hidden_gain = 1

    # Calculating gain for xavier init of output layer
    if 'output_activation' in list(module_dict.keys()):
        if module_dict['output_activation'] == 'relu':
            out_gain = math.sqrt(2)
        if module_dict['output_activation'] == 'tanh':
            out_gain = 5.0 / 3
    else:
        out_gain = 1

    kernel_init_type = module_dict['layer_kernel_init']
    bias_init_type = module_dict['layer_bias_init']

    for i, layer in enumerate(layers):
        if '_' in kernel_init_type:
            scheme, distribution = kernel_init_type.split('_')[0], kernel_init_type.split('_')[1]
            # Use only with relu or leaky_relu
            if scheme == 'kaiming' and distribution == 'uniform':
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in')
            elif scheme == 'kaiming' and distribution == 'normal':
                nn.init.kaiming_normal_(layer.weight, mode='fan_in')
            elif scheme == 'xavier' and distribution == 'uniform':
                if i == module_dict['hidden_layers']:
                    nn.init.xavier_uniform_(layer.weight, gain=out_gain)
                else:
                    nn.init.xavier_uniform_(layer.weight, gain=hidden_gain)
            else:
                if i == module_dict['hidden_layers']:
                    nn.init.xavier_normal_(layer.weight, gain=out_gain)
                else:
                    nn.init.xavier_normal_(layer.weight, gain=hidden_gain)
        elif kernel_init_type[i] == 'normal':
            nn.init.normal_(layer.weight, mean=0, std=1)
        elif kernel_init_type[i] == 'uniform':
            nn.init.uniform_(layer.weight, a=0, b=1)
        else:
            raise ValueError(' --> Provided weight init scheme not among defined kernel init schemes !')

    # Initialize bias from one of the simple initialization schemes
    for i, layer in enumerate(layers):
        if '_' in bias_init_type:
            scheme, distribution = bias_init_type.split('_')[0], bias_init_type.split('_')[1]
            # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
            if scheme == 'xavier' and distribution == 'uniform':
                if i == module_dict['hidden_layers']:
                    nn.init.xavier_uniform_(layer.bias, gain=out_gain)
                else:
                    nn.init.xavier_uniform_(layer.bias, gain=hidden_gain)
            else:
                if i == module_dict['hidden_layers']:
                    nn.init.xavier_normal_(layer.bias, gain=out_gain)
                else:
                    nn.init.xavier_normal_(layer.bias, gain=hidden_gain)
        elif bias_init_type == 'zeros':
            nn.init.zeros_(layer.bias)
        elif bias_init_type == 'uniform':
            in_dim = layer.weight.size(1)
            nn.init.uniform_(layer.bias, a=-math.sqrt(1/in_dim), b=math.sqrt(1/in_dim))
        else:
            raise ValueError(' --> Provided bias init scheme not among defined bias init schemes !')

    return layer_list

def set_layer_activation(activation):
    """Set layer activation function.

    Args:
        activation (str) : string specifying the activation function

    Returns : activation_obj (torch.nn.modules.activation)
    """
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.5)
    elif activation == 'elu':
        return nn.ELU(alpha=1.0)
    elif activation == 'prelu':
        return nn.PReLU(num_parameters=1, init=0.25)
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softmax':
        return nn.Softmax(dim=-1)
    elif activation == 'logsoftmax':
        return nn.LogSoftmax(dim=-1)
    elif activation == 'softsign':
        return nn.Softsign()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'hardtanh':
        return nn.Hardtanh(min_val=-2, max_val=2)
    elif activation == 'tanhshrink':
        return nn.Tanhshrink()
    elif activation == 'softplus':
        return nn.Softplus(beta=5, threshold=20)
    elif activation == 'silu':
        return nn.SiLU()
    else:
        raise ValueError(f' --> {activation} not among defined activation functions !')

def set_layer_dropout(dropout_type, p):
    """Set layer dropout.

    Args:
        dropout_type (str) : string specifying the dropout function
        p (float) : dropout probability
    
    Returns : dropout_obj (torch.nn.modules.dropout)
    """
    if dropout_type == 'Dropout':
        return nn.Dropout(p=p)
    elif dropout_type == 'AlphaDropout':
        return nn.AlphaDropout(p=p)
    else:
        raise ValueError(f' --> {dropout_type} not among defined dropout functions !')

def create_loss_object(loss):
    """Create loss object.

    Args : 
        loss (str) : a string specifying the loss function
        
    Returns : loss_obj (torch.nn.modules.loss)
    """
    if loss == 'mae':
        return nn.L1Loss()
    elif loss == 'mse':
        return nn.MSELoss()
    elif loss == 'huber':
        return nn.HuberLoss()
    elif loss == 'hinge':
        return nn.HingeEmbeddingLoss()
    elif loss == 'kld':
        return nn.KLDivLoss()
    elif loss == 'nll':
        return nn.NLLLoss(reduction='mean')
    elif loss == 'ce':
        return nn.CrossEntropyLoss(reduction='mean')
    elif loss == 'bce':
        # Reduction : Mean (The mean loss of a batch is calculated)
        # Reduction : Sum (The loss is summed over the batch)
        return nn.BCELoss(reduction='mean')
    elif loss == 'bcewithlogits':
        # Reduction : Mean (The mean loss of a batch is calculated)
        # Reduction : Sum (The loss is summed over the batch)
        return nn.BCEWithLogitsLoss(reduction='mean')
    elif loss == 'name_of_loss':
        #tf_loss = nameOfLoss()
        raise ValueError(' --> Loss Not Implemented !')
    else:
        raise ValueError(f' --> {loss} not among defined losses !')
    
def create_metric_object(metric, num_classes=None):
    """Create metric object.
    
    Args:
        metric (str) : a string specifying the metric function

    Returns : metric_obj 
    """
    # NOTE :
    # -> Best result is 0. Bad predictions can lead to arbitrarily large values.
    # -> This occurs when the target is close to 0. MAPE returns a large number instead of inf
    if metric == 'mape':
        return MeanAbsolutePercentageError()
    elif metric=='rmse':
        return MeanSquaredError(squared=False)
    elif metric == 'mse':
        return MeanSquaredError(squared=True)
    elif metric == 'mae':
        return MeanAbsoluteError()
    elif metric == 'ce':
        return Accuracy(task="multilabel", num_labels=num_classes, average='macro')
    elif metric == 'bcewithlogits':
        metric = Accuracy(task="binary", threshold=0.5)
    else:
        raise NotImplementedError(f' --> {metric} not among defined metrics !')

def create_scheduler_object(optimizer, nn_train_params_dict):
    """Create scheduler.
    
    Args:
        optimizer : optimizer object
        nn_train_params_dict : dictionary containing training parameters

    Returns : lr_scheduler_config (dict) : dictionary containing scheduler configuration
    """
    if nn_train_params_dict['scheduler']['type'] == 'expo':
        gamma = nn_train_params_dict['scheduler']['gamma']
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                                gamma=gamma,
                                                                verbose=True)
    if nn_train_params_dict['scheduler']['type'] == 'step':
        step_size = nn_train_params_dict['scheduler']['step_size']
        gamma = nn_train_params_dict['scheduler']['gamma'] 
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                        step_size=step_size,
                                                        gamma=gamma,
                                                        verbose=True)
    if nn_train_params_dict['scheduler']['type'] == 'multi_step':
        milestones = nn_train_params_dict['scheduler']['milestones']
        gamma = nn_train_params_dict['scheduler']['gamma']        
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=milestones,
                                                            gamma=gamma,
                                                            verbose=True)
    if nn_train_params_dict['scheduler']['type'] == 'reduce_lr_on_plateau':
        mode = nn_train_params_dict['scheduler']['mode']
        factor = nn_train_params_dict['scheduler']['factor']
        patience = nn_train_params_dict['scheduler']['patience']
        cooldown = nn_train_params_dict['scheduler']['cooldown']
        min_lr = nn_train_params_dict['scheduler']['min_lr']
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                    mode=mode,
                                                                    factor=factor,
                                                                    patience=patience,
                                                                    cooldown=cooldown,
                                                                    min_lr=min_lr,
                                                                    verbose=True)   
    else:
        raise ValueError(' --> Provided Learning Rate Scheduling scheme has not been defined.')
    lr_scheduler_config = {# REQUIRED: The scheduler instance
                            "scheduler": lr_scheduler,
                            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                            "monitor": nn_train_params_dict['scheduler']['monitor'],
                            "frequency": nn_train_params_dict['scheduler']['frequency'],
                            # If set to `True`, will enforce that the value specified 'monitor'
                            # is available when the scheduler is updated, thus stopping
                            # training if not found. If set to `False`, it will only produce a warning
                            "strict": True,
                            # If using the `LearningRateMonitor` callback to monitor the
                            # learning rate progress, this keyword can be used to specify
                            # a custom logged name
                            "name": nn_train_params_dict['scheduler']['name']}
    return lr_scheduler_config

# Class that creates the TF optimizer object            
def create_optimizer_object(modules, nn_train_params_dict):
    """Create optimizer.
    
    Args:
        modules (torch.nn.ModuleDict()) : dictionary containing torch modules
        nn_train_params_dict : dictionary containing training parameters

    Returns : optimizer_object (torch.optim)
    """
    optimizer_type = nn_train_params_dict['optimizer']['type']
    # Check if per module lr is required 
    if check_dict_key_exists('module_name', nn_train_params_dict['optimizer']):
        params_groups = []
        for i, module_name in enumerate(nn_train_params_dict['optimizer']['module_name']):
            module_params = {}
            module_params['params'] = modules[module_name].parameters()
            module_params['lr'] = nn_train_params_dict['optimizer']['lr'][i]
            params_groups.append(module_params)
        if optimizer_type == 'adam':
            adam_optimizer = optim.Adam(params_groups,
                                        betas=(0.9, 0.999),
                                        eps=1e-8,
                                        weight_decay=0,
                                        amsgrad=False)
            return adam_optimizer
        elif optimizer_type == 'sgd':
            sgd_optimizer = optim.SGD(params_groups,
                                      momentum=nn_train_params_dict['optimizer']['momentum'])
            return sgd_optimizer
        else:
            raise ValueError(f' --> {optimizer_type} has not been defined.')
    else: # One optimizer for all parameters
        module_params = modules.parameters()
        if optimizer_type == 'adam':
            adam_optimizer = optim.Adam(module_params,
                                        lr=nn_train_params_dict['optimizer']['lr'],
                                        betas=(0.9, 0.999),
                                        eps=1e-8,
                                        weight_decay=0,
                                        amsgrad=False)
            return adam_optimizer
        elif optimizer_type == 'sgd':
            sgd_optimizer = optim.SGD(module_params,
                                      lr=nn_train_params_dict['optimizer']['lr'],
                                      momentum=nn_train_params_dict['optimizer']['momentum'])
            return sgd_optimizer
        else:
            raise ValueError(f' --> {optimizer_type} has not been defined.')

def create_callback_object(nn_train_params_dict, nn_save_dir):
    """Create callback.
    
    Args:
        nn_train_params_dict (dict) : dictionary containing training parameters
        nn_save_dir (str) : directory to save model checkpoints

    Returns:
        callback_objects (list) : list of callback objects
    """
    callback_objects = []
    callback_types = list(nn_train_params_dict['callbacks'].keys())
    callback_dicts = list(nn_train_params_dict['callbacks'].values())
    for i, callback_dict in enumerate(callback_dicts):
        callback_type = callback_types[i]
        if callback_type == 'early_stopping':
            early_stopping = EarlyStopping(monitor=callback_dict['monitor'],
                                            min_delta=callback_dict['min_delta'],
                                            patience=callback_dict['patience'],
                                            verbose=True,
                                            mode=callback_dict['mode'],
                                            strict=True,
                                            check_finite=True)
            callback_objects.append(early_stopping)
        elif callback_type == 'model_checkpoint':
            checkpoints_dir = nn_save_dir + '/checkpoints'
            model_checkpoint = ModelCheckpoint(dirpath=checkpoints_dir,
                                                filename='{epoch}-{total_val_loss:.2f}',
                                                monitor=callback_dict['monitor'],
                                                verbose=False,
                                                save_last=True,
                                                save_top_k=callback_dict['save_top_k'],
                                                mode=callback_dict['mode'],
                                                auto_insert_metric_name=True)
            callback_objects.append(model_checkpoint)
        elif callback_type == 'rich_model_summary':
            callback_objects.append(RichModelSummary(max_depth=1))
        elif callback_type == 'rich_progress_bar':
            progress_bar = RichProgressBar(
                            theme=RichProgressBarTheme(
                                description="green_yellow",
                                progress_bar="green1",
                                progress_bar_finished="green1",
                                progress_bar_pulse="#6206E0",
                                batch_progress="green_yellow",
                                time="grey82",
                                processing_speed="grey82",
                                metrics="grey82"),
                            leave=False)
            callback_objects.append(progress_bar)
        else:
            raise ValueError(f' --> {callback_type} callback not defined.')
    return callback_objects