" VanillaAE class "

import sys
import os
import pprint

import numpy as np
# Pytorch libraries
import torch
from torch import nn
from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError
import pytorch_lightning as pl

# User defined libraries
from .utils import nn_utils

class AE(pl.LightningModule):
    """AE class"""

    def __init__(self,
                 run_dir,
                 ae_save_dir,
                 run_id,
                 nn_params_dict,
                 nn_train_params_dict,
                 nn_datasets_dict):

        super(AE, self).__init__()

        # Save the parameters passed into __init__
        self.save_hyperparameters()
        self.automatic_optimization = True
        self.name = nn_params_dict['name']
        self.run_dir = run_dir
        self.ae_save_dir = ae_save_dir
        self.run_id = run_id
        self.global_seed = nn_train_params_dict['global_seed']
        self.nn_params_dict = nn_params_dict
        self.nn_train_params_dict = nn_train_params_dict
        self.nn_datasets_dict = nn_datasets_dict
        dataset_name = list(nn_datasets_dict['train'].keys())[0]
        self.datasets = torch.load(
            f'../runs/{run_dir}/{ae_save_dir}/datasets/{dataset_name}_dataset.pt')
        self.all_samples = self.datasets[:]
        self.example_input = self.datasets[0]
        self.submodule_dicts = nn_params_dict['submodules']
        self.submodule_losses = None
        # Create a dictionary to store all submodules
        self.submodules = torch.nn.ModuleDict()
        self.trace_model = False
        # Init variables to store batch and epoch loss results
        self.train_loss_values = None
        self.val_loss_values = None
        self.train_losses_batch = None
        self.val_losses_batch = None
        self.train_losses_epoch = None
        self.val_losses_epoch = None
        # Save outputs on epoch to pickle
        self.save_outputs_on_epoch = {}

        # Outer loop iterates over the submodules
        for submodule_name, submodule_dict in \
                zip(self.submodule_dicts.keys(), self.submodule_dicts.values()):

            # If building a variational autoencoder then 'z' is a special submodule
            if submodule_name == 'z':
                layer_list = None
            else:
                layer_list = torch.nn.ModuleList()

                for layer_num in range(submodule_dict['hidden_layers']):
                    # If layer type is linear then get input dimension
                    if submodule_dict['layer_type'] == 'linear':
                        if layer_num == 0:
                            # Calculate the input dimensions to first layer
                            input_dim = nn_utils.get_module_input_dim(submodule_dict['connect_to'],
                                                                      self.nn_params_dict,
                                                                      self.datasets.variable_shapes)
                            layer_list.append(nn.Linear(in_features=input_dim,
                                                        out_features=submodule_dict['hidden_dim'],
                                                        bias=True))
                        else:
                            layer_list.append(nn.Linear(in_features=submodule_dict['hidden_dim'],
                                                        out_features=submodule_dict['hidden_dim'],
                                                        bias=True))
                        # Add hidden activations if specified
                        if 'hidden_activation' in list(submodule_dict.keys()):
                            layer_list.append(nn_utils.set_layer_activation(
                                submodule_dict['hidden_activation']))
                        # Add dropout after each layer if specified
                        if 'layer_dropout' in list(submodule_dict.keys()):
                            dropout_type = submodule_dict['layer_dropout']['type']
                            p = submodule_dict['layer_dropout']['p']
                            layer_list.append(
                                nn_utils.set_layer_dropout(dropout_type, p))
                    elif submodule_dict['layer_type'] == 'conv2d':
                        raise NotImplementedError(
                            ' --> Conv2D layer not implemented yet.')
                    else:
                        raise ValueError(' --> Unknown layer type.')

                # Add the output layer
                layer_list.append(nn.Linear(in_features=submodule_dict['hidden_dim'],
                                            out_features=submodule_dict['output_dim'],
                                            bias=True))
                # Add output activations if specified
                if 'output_activation' in list(submodule_dict.keys()):
                    layer_list.append(nn_utils.set_layer_activation(
                        submodule_dict['output_activation']))

                # Initialize weights for all layers
                layer_list = nn_utils.set_layer_init(
                    layer_list, submodule_dict)

            # Check to see if a submodule has to be loaded
            if 'load_params' in list(submodule_dict.keys()):
                path = submodule_dict['load_params']
                try:
                    submodule_params = torch.load(path)
                    layer_list.load_state_dict(submodule_params['state_dict'])
                    print(f' --> Loaded submodule {submodule_name}.')
                except OSError as err:
                    raise FileNotFoundError(
                        f' --> Could not load {submodule_name}.') from err

            # Finally add to submodule list
            self.submodules.update({submodule_name: layer_list})

            print('\n')
            print(f' --> Submodule {submodule_name} layers :')
            print(layer_list)

    def forward(self, module_inputs):
        """Forward pass through the model."""

        # Stores all submodule outputs
        submodule_outputs = {}

        module_input_ids = list(module_inputs.keys())

        # Outer loop iterates over the submodules
        for submodule_name, submodule in \
                zip(self.submodules.keys(), self.submodules.values()):

            # Get the input ids that are connected to the submodule
            submodule_input_ids = self.submodule_dicts[submodule_name]['connect_to']

            # Case : input sampled from a probability distribution
            if submodule_name == 'z':
                if self.submodule_dicts['z']['sample_from'] == 'normal':
                    # Required submoduel keys
                    mu, logvar = submodule_outputs['mu'], submodule_outputs['logvar']
                    inp = [mu, logvar]
                    std = logvar.exp().mul(0.5)
                    eps = torch.randn_like(std)
                    output = eps*std + mu
                else:
                    raise ValueError('Reparameterization from chosen distribution not defined.\
                                    Please add Reparameterization scheme in forward().')
            # Case : input comes from a dataset or another submodule
            else:
                inp = []
                for submodule_input_id in submodule_input_ids:
                    if submodule_input_id in module_input_ids:
                        inp.append(module_inputs[submodule_input_id])
                    else:  # Then input is output from previous module
                        inp.append(submodule_outputs[submodule_input_id])
                # Convert list of tensors to a single tensor
                inp = torch.concatenate(inp, dim=-1)
                for j, layer in enumerate(submodule):
                    if j == 0:
                        output = layer(inp)
                    else:
                        output = layer(output)
            submodule_outputs[submodule_name] = output

            if self.trace_model:
                print('\n')
                print(' ---------------------------------- ')
                print(f'module_name:{submodule_name}')
                print(f'input id:{submodule_input_ids}')
                print('input to submodule :')
                print(inp)
                print(f'output id:{submodule_name}')
                print('output from submodule :')
                print(output)
                print('Submodule output dictionary :')
                pp = pprint.PrettyPrinter()
                pp.pprint(submodule_outputs)
                print(' ---------------------------------- ')
                print('\n')

        return submodule_outputs

    # <-- TESTING -->
    # Uncomment for using custom pytorch training loops
    # def backward(self, loss, optimizer):
    #    loss.backward()

    def compile(self):
        """Compile the model."""
        losses = {}
        for submodule_name, submodule_dict in \
                zip(self.submodule_dicts.keys(), self.submodule_dicts.values()):
            if 'loss' in list(submodule_dict.keys()):
                losses[submodule_name] = nn_utils.create_loss_object(
                    submodule_dict['loss']['type'])
        self.submodule_losses = losses

    def configure_optimizers(self):
        """Configure the optimizers."""
        optimizer = nn_utils.create_optimizer_object(
            self.submodules, self.nn_train_params_dict)
        if 'scheduler' in list(self.nn_train_params_dict.keys()):
            scheduler = nn_utils.create_scheduler_object(
                optimizer, self.nn_train_params_dict)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return {'optimizer': optimizer}

    def training_step(self, batch, batch_idx):
        """Training step."""
        # <-- TESTING -->
        # torch.set_grad_enabled(True)
        # Call the  optimizers
        # opt = self.optimizers()
        # opt.zero_grad()

        # Squeeze all the tensors
        # for dataset in batch.keys():
        #    for variable in batch[dataset].keys():
        #        batch[dataset][variable] = torch.squeeze(batch[dataset][variable])

        # Pass data into model
        submodule_outputs = self(batch)

        train_loss_values = {}

        # Init total loss to 0
        total_train_loss = torch.tensor(
            0, device=self.device, dtype=torch.float32)

        # Prediction losses
        for submodule_name, submodule_loss in \
                zip(self.submodule_losses.keys(), self.submodule_losses.values()):

            # Output from submodule
            output = submodule_outputs[submodule_name]

            variable_name = self.submodule_dicts[submodule_name]['loss']['target']
            target = batch[variable_name]

            loss_type = self.submodule_dicts[submodule_name]['loss']['type']
            loss_wt = self.submodule_dicts[submodule_name]['loss']['wt']
            loss_wt = torch.tensor(
                loss_wt, device=self.device, dtype=torch.float32, requires_grad=False)

            if 'sample_wts' in list(batch.keys()):
                sample_wt = batch['sample_wts']
            else:
                sample_wt = torch.tensor(
                    1, device=self.device, dtype=torch.float32, requires_grad=False)

            pred_loss = submodule_loss(output, target).multiply(
                loss_wt).multiply(sample_wt)

            total_train_loss = total_train_loss.add(pred_loss)

            # NOTE :
            # -> Best result is 0. Bad predictions can lead to arbitrarily large values.
            # -> This occurs when the target is close to 0. MAPE returns a large number instead of inf
            if loss_type == 'mae' or loss_type == 'huber':
                # metric = MeanAbsolutePercentageError()
                # metric_name = 'mape'
                # metric_train = metric(output, target)
                metric = MeanSquaredError(squared=False)
                metric_name = 'rmse'
                metric_train = metric(output, target)
            elif loss_type == 'mse':
                metric = MeanAbsoluteError()
                metric_name = 'mae'
                metric_train = metric(output, target)
            elif loss_type == 'ce':
                num_classes = target.size(1)
                metric = Accuracy(task="multilabel",
                                  num_labels=num_classes, average='macro')
                metric_name = 'accuracy'
                # output = torch.nn.LogSoftmax(dim=-1)(output)
                metric_train = metric(output, target)
            elif loss_type == 'bcewithlogits':
                metric = Accuracy(task="binary", threshold=0.5)
                metric_name = 'binary_accuracy'
                # output = torch.nn.LogSoftmax(dim=-1)(output)
                metric_train = metric(output, target)
            else:
                raise NotImplementedError(
                    f' --> Loss type {loss_type} not implemented.')

            # The weighted Pred loss
            train_loss_values['train_' + variable_name +
                              '_' + loss_type] = pred_loss.item()
            train_loss_values['train_' + variable_name +
                              '_' + metric_name] = metric_train.item()

        # <-- USER START -->
        # Can define custom losses here
        # Make sure to log your loss after the total loss is logged so
        # that plot_utils.py -u metrics can create the appropriate train and val prediction loss plots

        # Access the outputs of any hidden layer by : output = submodule_outputs['layer_name']

        compute_kld_loss = False

        # KLD Loss for variational autoencoder
        if compute_kld_loss:

            if self.submodule_dicts['z']['sample_from'] == 'normal':
                mu = submodule_outputs['mu']
                logvar = submodule_outputs['logvar']

                # torch.sum() calculates the kld_loss for a single sample e
                # torch.mean() calculated the mean kld_loss over mini batch
                kld_loss = torch.mean(-0.5*torch.sum(1 + logvar -
                                      mu.pow(2) - logvar.exp(), dim=1), dim=0)

            # Beta VAE : https://openreview.net/forum?id=Sy2fzU9gl
            beta = torch.tensor(1, device=self.device,
                                dtype=torch.float32, requires_grad=False)

            if 'sample_wts' in list(batch.keys()):
                sample_wt = batch['sample_wts']['wts']
                sample_wt = torch.tensor(
                    sample_wt, device=self.device, dtype=torch.float32, requires_grad=False)
            else:
                sample_wt = torch.tensor(
                    1, device=self.device, dtype=torch.float32, requires_grad=False)

            kld_loss = kld_loss.multiply(beta).multiply(sample_wt)

            total_train_loss = total_train_loss.add(kld_loss)

        # <-- USER END -->

        # Init Regularization losses
        l1_param_loss = torch.tensor(
            0, device=self.device, dtype=torch.float32)
        l2_param_loss = torch.tensor(
            0, device=self.device, dtype=torch.float32)

        # Regularization losses
        for submodule_name, submodule_dict in \
                zip(self.submodule_dicts.keys(), self.submodule_dicts.values()):
            if 'layer_weight_reg_l1' in list(submodule_dict.keys()):
                lambda_l1 = submodule_dict['layer_weight_reg_l1']
                lambda_l1 = torch.tensor(
                    lambda_l1, device=self.device, dtype=torch.float32, requires_grad=False)
                for name, params in self.submodules[submodule_name].named_parameters():
                    if name.endswith('.weight') or name.endswith('.bias'):
                        p = params.view(-1)
                        l1_param_loss = l1_param_loss.add(
                            p.abs().sum().multiply(lambda_l1))
                lambda_l2 = submodule_dict['layer_weight_reg_l2']
                lambda_l2 = torch.tensor(
                    lambda_l2, device=self.device, dtype=torch.float32, requires_grad=False)
                for name, params in self.submodules[submodule_name].named_parameters():
                    if name.endswith('.weight') or name.endswith('.bias'):
                        # Params view is important here since weights is a 2D tensor which we unwrap to a 1D tensor
                        # params.data :- returns the weight data. No reshape
                        # params.view :- returns the weight data. With reshape
                        p = params.view(-1)
                        l2_param_loss = l2_param_loss.add(
                            p.pow(2).sum().multiply(lambda_l2))

        # Add in regularization losses
        total_train_loss += l1_param_loss + l2_param_loss
        # The weighted loss
        train_loss_values['total_train_loss'] = total_train_loss.item()
        # KLD loss
        if compute_kld_loss:
            train_loss_values['kld_loss'] = kld_loss.item()

        train_loss_values['l1_param_loss'] = l1_param_loss.item()
        train_loss_values['l2_param_loss'] = l2_param_loss.item()

        self.train_loss_values = train_loss_values
        self.log_dict(train_loss_values, on_step=False,
                      on_epoch=True, logger=True, prog_bar=False,
                      batch_size=self.nn_train_params_dict['batch_size'])

        return total_train_loss

        # <-- USER START -->

        # Manually backward total loss
        # self.manual_backward(total_train_loss)
        # opt.step()
        # if self.lr_schedulers() != None:
        #    sch = self.lr_schedulers()
        #    sch.step()
        # torch.set_grad_enabled(False)

        # <-- USER END -->

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Pass data into model
        submodule_outputs = self(batch)
        # Init total loss to 0
        total_val_loss = torch.tensor(
            0, device=self.device, dtype=torch.float32)
        val_loss_values = {}
        # Prediction losses
        for submodule_name, submodule_loss in \
                zip(self.submodule_losses.keys(), self.submodule_losses.values()):
            # Output from submodule
            output = submodule_outputs[submodule_name]
            variable_name = self.submodule_dicts[submodule_name]['loss']['target']
            target = batch[variable_name]
            loss_type = self.submodule_dicts[submodule_name]['loss']['type']
            loss_wt = self.submodule_dicts[submodule_name]['loss']['wt']
            loss_wt = torch.tensor(
                loss_wt, device=self.device, dtype=torch.float32, requires_grad=False)
            if 'sample_wts' in list(batch.keys()):
                sample_wt = batch['sample_wts']['wts']
                sample_wt = torch.tensor(
                    sample_wt, device=self.device, dtype=torch.float32)
            else:
                sample_wt = torch.tensor(
                    1, device=self.device, dtype=torch.float32)
            pred_loss = submodule_loss(output, target)*loss_wt*sample_wt
            total_val_loss += pred_loss
            if loss_type == 'mae' or loss_type == 'huber':
                metric = MeanSquaredError(squared=False)
                metric_name = 'rmse'
                metric_val = metric(output, target)
                # metric = MeanAbsolutePercentageError()
                # metric_name = 'mape'
                # metric_val = metric(output, target)
            elif loss_type == 'mse':
                metric = MeanAbsoluteError()
                metric_name = 'mae'
                metric_val = metric(output, target)
            elif loss_type == 'ce':
                num_classes = target.size(1)
                metric = Accuracy(task="multilabel",
                                  num_labels=num_classes, average='macro')
                metric_name = 'accuracy'
                metric_val = metric(output, target)
            elif loss_type == 'bcewithlogits':
                metric = Accuracy(task="binary", threshold=0.5)
                metric_name = 'binary_accuracy'
                # output = torch.nn.LogSoftmax(dim=-1)(output)
                metric_val = metric(output, target)
            else:
                raise NotImplementedError(
                    f' --> Loss type {loss_type} not implemented.')
            # The weighted loss
            val_loss_values['val_' + variable_name +
                            '_' + loss_type] = pred_loss.item()
            val_loss_values['val_' + variable_name +
                            '_' + metric_name] = metric_val.item()
        val_loss_values['total_val_loss'] = total_val_loss.item()
        self.val_loss_values = val_loss_values
        self.log_dict(val_loss_values,
                      on_step=False,
                      on_epoch=True,
                      logger=True,
                      prog_bar=False,
                      batch_size=self.nn_train_params_dict['batch_size'])

    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        return self(batch)

    def test_step(self, batch, batch_idx):
        """Test step."""

    ################################################################################################
    # Pytorch Lightning Model Hooks go here
    ################################################################################################

    def on_fit_start(self):
        """Called when fit begins."""

        # Show example of input to model
        print(' --> Example Input : ')
        print(self.example_input)
        print('\n')

        print('--> Model Trace : ')
        self.trace_model = True
        # Check model hierarchy
        module_out = self(self.example_input)
        self.trace_model = False

        sys.stdout.close()
        sys.stdout = sys.__stdout__

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Store the submodule outputs
        submodule_outputs = self(self.all_samples)
        for submodule_name, submodule_dict in \
                zip(self.submodule_dicts.keys(), self.submodule_dicts.values()):
            if 'save_output_on_epoch_end' in list(submodule_dict.keys()):
                if submodule_dict['save_output_on_epoch_end'] is True:
                    submodule_output = submodule_outputs[submodule_name]
                    if self.current_epoch == 0:
                        self.save_outputs_on_epoch[submodule_name] = [
                            submodule_output]
                    else:
                        self.save_outputs_on_epoch[submodule_name].append(
                            submodule_output)

    def on_fit_end(self):
        """Called at the end of fit() to do things such as logging, saving etc."""

        for submodule_name, submodule_dict in \
                zip(self.submodule_dicts.keys(), self.submodule_dicts.values()):

            if 'save_output_on_fit_end' in list(submodule_dict.keys()):
                if submodule_dict['save_output_on_fit_end'] is True:
                    # Create a submodule outputs dir if it does not exist
                    submodule_outputs_dir = f'../runs/{self.run_dir}/{self.ae_save_dir}/ae_param_search/{self.run_id}/submodule_outputs'
                    if not os.path.exists(submodule_outputs_dir):
                        os.mkdir(submodule_outputs_dir)
                    if not os.path.exists(submodule_outputs_dir + '/train'):
                        os.mkdir(submodule_outputs_dir + '/train')
                    submodule_outputs = self(self.all_samples)
                    submodule_output = submodule_outputs[submodule_name]
                    submodule_output_arr = submodule_output.detach().numpy()
                    filename = submodule_name + '_output_on_fit_end.csv'
                    np.savetxt(submodule_outputs_dir + '/train/' + filename,
                            submodule_output_arr,
                            delimiter=',')

            if 'save_output_on_epoch_end' in list(submodule_dict.keys()):
                if submodule_dict['save_output_on_epoch_end'] is True:
                    # Create a submodule outputs dir if it does not exist
                    submodule_outputs_dir = f'../runs/{self.run_dir}/{self.ae_save_dir}/ae_param_search/{self.run_id}/submodule_outputs'
                    if not os.path.exists(submodule_outputs_dir):
                        os.mkdir(submodule_outputs_dir)
                    if not os.path.exists(submodule_outputs_dir + '/train'):
                        os.mkdir(submodule_outputs_dir + '/train')
                    # Save the 3D numpy array to a pickle in submodule outputs dir
                    for submodule_name, submodule_output_on_epoch in \
                            zip(self.save_outputs_on_epoch.keys(), self.save_outputs_on_epoch.values()):
                        submodule_output_3D = torch.stack(
                            (submodule_output_on_epoch))
                        pickle_path = submodule_outputs_dir + '/train/' + \
                            submodule_name + '_output_on_epoch_end.pt'
                        torch.save(submodule_output_3D, pickle_path)

            if 'save_params' in list(submodule_dict.keys()):
                if submodule_dict['save_params'] is True:
                    # Make a directory to store the submodules
                    submodules_dir = f'../runs/{self.run_dir}/{self.ae_save_dir}/ae_param_search/{self.run_id}/submodule_params'
                    if not os.path.exists(submodules_dir):
                        os.mkdir(submodules_dir)
                    submodule_path = f'{submodules_dir}/{submodule_name}_params.pt'
                    torch.save({'state_dict': self.submodules[submodule_name].state_dict()},
                            submodule_path)


if __name__ == '__main__':
    run_dir = 'test_run'
    ae_save_dir = 'test_ae'
    run_id = 'test_runid'
    from inputs_perov_data.nn_inputs import list_of_nn_params_dict
    from inputs_perov_data.train_inputs import list_of_nn_train_params_dict
    from inputs_perov_data.dataset_inputs import list_of_nn_datasets_dict
    nn_params_dict = list_of_nn_params_dict[0]
    nn_train_params_dict = list_of_nn_train_params_dict[0]
    nn_datasets_dict = list_of_nn_datasets_dict[0]
    test_ae = AE(run_dir, ae_save_dir, run_id, nn_params_dict,
                 nn_train_params_dict, nn_datasets_dict)
    # Model forward propagation
    submodule_outputs = test_ae(test_ae.all_samples)