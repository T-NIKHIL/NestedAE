" Contain implementations of different machine learning models "

# Pytorch libraries
import torch
from torch import nn
from torchmetrics import MeanAbsolutePercentageError, Accuracy, MeanSquaredError
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import pprint
import sys
import os

custom_arch = False
# If true mention the feature dimensions for each input in the decoder.
multiple_outputs = False

activation_fns = {'relu': torch.nn.ReLU(), 
                  'elu': torch.nn.ELU(),
                  'tanh': torch.nn.Tanh(), 
                  'sigmoid': torch.nn.Sigmoid(),
                  'softmax': torch.nn.Softmax(dim=1),
                   None: None}

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        try:
            self.activation_fn = activation_fns[activation_fn]
        except KeyError:
            raise ValueError('Invalid activation function')

        self.layers = torch.nn.ModuleList()

        if self.num_layers == 1:
            self.layers.append(torch.nn.Linear(self.input_dim, self.latent_dim))
            if self.activation_fn is not None:
                self.layers.append(self.activation_fn)
        else:
            for i in range(self.num_layers):
                if i == 0:
                    self.layers.append(torch.nn.Linear(self.input_dim, self.hidden_dim))
                    if self.activation_fn is not None:
                        self.layers.append(self.activation_fn)
                    self.layers.append(torch.nn.Dropout(self.dropout))
                elif i == self.num_layers - 1:
                    self.layers.append(torch.nn.Linear(self.hidden_dim, self.latent_dim))
                else:
                    self.layers.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                    if self.activation_fn is not None:
                        self.layers.append(self.activation_fn)
                    self.layers.append(torch.nn.Dropout(self.dropout))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                z = layer(x)
            else:
                z = layer(z)
        return z
    
class Predictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn, output_activation_fn):
        super(Predictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        try:
            self.activation_fn = activation_fns[activation_fn]
            self.output_activation_fn = activation_fns[output_activation_fn]
        except KeyError:
            raise ValueError('Invalid activation function')
        self.custom_arch = False
    
        self.layers = torch.nn.ModuleList()
        
        if custom_arch:
            self.layers.append(torch.nn.Linear(self.latent_dim, self.hidden_dim))
            self.layers.append(torch.nn.Tanh())
            self.layers.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(torch.nn.Tanh())
            self.layers.append(torch.nn.Linear(self.hidden_dim, 1))
        else:
            if self.num_layers == 1:
                self.layers.append(torch.nn.Linear(self.latent_dim, 1))
                if self.output_activation_fn is not None:
                    self.layers.append(self.output_activation_fn)
            else:
                for i in range(self.num_layers):
                    if i == 0:
                        self.layers.append(torch.nn.Linear(self.latent_dim, self.hidden_dim))
                        if self.activation_fn is not None:
                            self.layers.append(self.activation_fn)
                        self.layers.append(torch.nn.Dropout(self.dropout))
                    elif i == self.num_layers - 1:
                        self.layers.append(torch.nn.Linear(self.hidden_dim, 1))
                        if self.output_activation_fn is not None:
                            self.layers.append(self.output_activation_fn)
                    else:
                        self.layers.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                        if self.activation_fn is not None:
                            self.layers.append(self.activation_fn)
                        self.layers.append(torch.nn.Dropout(self.dropout))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                pred = layer(x)
            else:
                pred = layer(pred)
        return pred
    
class Decoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn, output_activation_fn):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        try:
            self.activation_fn = activation_fns[activation_fn]
            self.output_activation_fn = activation_fns[output_activation_fn]
        except KeyError:
            raise ValueError('Invalid activation function')
        self.multiple_outputs = False
        
        self.layers = torch.nn.ModuleList()

        if self.num_layers == 1:
            if self.multiple_outputs:
                self.layers.append(torch.nn.Linear(self.latent_dim, self.hidden_dim))
                if self.output_activation_fn is not None:
                    self.layers.append(self.output_activation_fn)
            else:
                self.layers.append(torch.nn.Linear(self.latent_dim, self.input_dim))
                if self.output_activation_fn is not None:
                    self.layers.append(self.output_activation_fn)

        else:
            for i in range(self.num_layers):
                if i == 0:
                    self.layers.append(torch.nn.Linear(self.latent_dim, self.hidden_dim))
                    if self.activation_fn is not None:
                        self.layers.append(self.activation_fn)
                    self.layers.append(torch.nn.Dropout(self.dropout))
                elif i == self.num_layers - 1:
                    if self.multiple_outputs:
                        self.layers.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                        if self.output_activation_fn is not None:
                            self.layers.append(self.output_activation_fn)
                    else:
                        self.layers.append(torch.nn.Linear(self.hidden_dim, self.input_dim))
                        if self.output_activation_fn is not None:
                            self.layers.append(self.output_activation_fn)
                else:
                    self.layers.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                    if self.activation_fn is not None:
                        self.layers.append(self.activation_fn)
                    self.layers.append(torch.nn.Dropout(self.dropout))

        if self.multiple_outputs:
            self.output1_layer = torch.nn.Linear(self.hidden_dim, 3)
            self.output2_layer = torch.nn.Linear(self.hidden_dim, 4)  
            self.output3_layer = torch.nn.Linear(self.hidden_dim, 8)      
            self.output4_layer = torch.nn.Linear(self.hidden_dim, 8)
            # self.output1_layer = torch.nn.Linear(self.hidden_dim, 15)
            # self.output2_layer = torch.nn.Linear(self.hidden_dim, 4)  

            
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                recon = layer(x)
            else:
                recon = layer(recon)
        if self.multiple_outputs:
            recon1 = self.output1_layer(recon)
            recon2 = self.output2_layer(recon)
            recon3 = self.output3_layer(recon)
            recon4 = self.output4_layer(recon)
            return recon1, recon2, recon3, recon4
            # return recon1, recon2
        else:
            return recon

class SupervisedSimpleAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn, pred_activation_fn, dec_activation_fn):
        super(SupervisedSimpleAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn)
        self.predictor = Predictor(input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn, pred_activation_fn)
        self.decoder = Decoder(input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn, dec_activation_fn)
        # If multiple outputs then define the layers here ...

    def forward(self, x):
        z = self.encoder(x)
        pred = self.predictor(z)
        if multiple_outputs:
            recon1, recon2 = self.decoder(z)
            return z, pred, recon1, recon2
        else:
            recon = self.decoder(z)
            return z, pred, recon
    
class UnsupervisedSimpleAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn, dec_activation_fn):
        super(UnsupervisedSimpleAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn)
        self.decoder = Decoder(input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn, dec_activation_fn)
        # If multiple outputs then define the layers here ...

    def forward(self, x):
        z = self.encoder(x)
        if multiple_outputs:
            recon1, recon2, recon3, recon4  = self.decoder(z)
            return z, recon1, recon2, recon3, recon4 
        else:
            recon = self.decoder(z)
            return z, recon
    
class SupervisedVAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn, pred_activation_fn, dec_activation_fn):
        super(SupervisedVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn)
        self.predictor = Predictor(input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn, pred_activation_fn)
        self.decoder = Decoder(input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn, dec_activation_fn)
        self.mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar = torch.nn.Linear(hidden_dim, latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        pred = self.predictor(z)
        reconst = self.decoder(z)
        return z, pred, reconst, mu, logvar
    
class UnsupervisedVAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn, dec_activation_fn):
        super(UnsupervisedVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn)
        self.decoder = Decoder(input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn, dec_activation_fn)
        self.mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar = torch.nn.Linear(hidden_dim, latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        reconst = self.decoder(z)
        return z, reconst, mu, logvar
    
# Check out these github repos for how to code the model and the loss function :

# 1. https://github.com/jariasf/GMVAE/tree/master
# 2. https://github.com/RuiShu/vae-clustering
    
class GMVAE(torch.nn.Module):
    def __init__(self, input_dim, y_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn, dec_activation_fn):
        super(GMVAE, self).__init__()
        """
        A GMVAE has three main modules:
        q(y|x) : Predict the class label based on X
        q(z|y,x) : Predict the latent variable based on X and the class label
        """
        self.encoder = Encoder(input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn)
        self.decoder = Decoder(input_dim, hidden_dim, dropout, latent_dim, num_layers, activation_fn, dec_activation_fn)
        try:
            self.activation_fn = activation_fns[activation_fn]
        except KeyError:
            raise ValueError('Invalid activation function')
        self.activation_fn = activation_fns[activation_fn]
        self.input_dim = input_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.qy_logit_x_layers = torch.nn.ModuleList()
        if self.num_layers == 1:
            self.qy_logit_x_layers.append(torch.nn.Linear(self.input_dim, self.y_dim))
            self.qy_logit_x_layers.append(self.activation_fn)
        else:
            for i in range(self.num_layers):
                if i == 0:
                    self.qy_logit_x_layers.append(torch.nn.Linear(self.input_dim, self.hidden_dim))
                    self.qy_logit_x_layers.append(self.activation_fn)
                    self.qy_logit_x_layers.append(torch.nn.Dropout(self.dropout))
                elif i == self.num_layers - 1:
                    self.qy_logit_x_layers.append(torch.nn.Linear(self.hidden_dim, self.y_dim))
                else:
                    self.qy_logit_x_layers.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                    self.qy_logit_x_layers.append(self.activation_fn)
                    self.qy_logit_x_layers.append(torch.nn.Dropout(self.dropout))

        self.mu = torch.nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.latent_dim),
            torch.nn.Softplus()
        )
        self.mu_prior = torch.nn.Linear(self.y_dim, self.latent_dim)
        self.logvar_prior = torch.nn.Sequential(
            torch.nn.Linear(self.y_dim, self.latent_dim),
            torch.nn.Softplus()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        for i, qy_logit_x_layer in enumerate(self.qy_logit_x_layers):
            if i == 0:
                qy_logit = qy_logit_x_layer(x)
            else:
                qy_logit = qy_logit_x_layer(qy_logit)
        qy = torch.nn.Softmax(dim=1)(qy_logit)

        # Defining a tensor that will store the fixed class label for all members of the batch
        y_ = torch.zeros([x.shape[0], self.y_dim])
        z, pred, reconst, mu, logvar, mu_prior, logvar_prior = [[None] * 10 for i in range(7)]
        for i in range(self.y_dim):
            # Add the class label to the tensor
            y = y_ + torch.eye(self.y_dim)[i]
            # Note to self : The generative model can take the predicted class label as input. This is what is done in the GMVAE repo
            # Note to self : In the Rui Shu repo the class label (y) is provided as a one hot vector. 
            h = torch.cat([x, y], dim=1)
            for j, encoder_layer in enumerate(self.encoder_layers):
                if j == 0:
                    h = encoder_layer(h)
                else:
                    h = encoder_layer(h)
            mu[i] = self.mu(h)
            logvar[i] = self.logvar(h)
            # Note to self : Can use the reparameterization trick here instead. This gives modified M2 in Rui Shu's repo.
            # Using the predicted mean and logvar sample from a gaussian distribution.
            # z[i] = torch.normal(mu[i], logvar[i].exp().sqrt())
            z[i] = self.reparameterize(mu[i], logvar[i])
            for j, pred_layer in enumerate(self.predictor_layers):
                if j == 0:
                    pred[i] = pred_layer(z[i])
                else:
                    pred[i] = pred_layer(pred[i])    
            mu_prior[i] = self.mu_prior(y)
            logvar_prior[i] = self.logvar_prior(y)
            for j, decoder_layer in enumerate(self.decoder_layers):
                if j == 0:
                    reconst[i] = decoder_layer(z[i])
                else:
                    reconst[i] = decoder_layer(reconst[i])
        return z, pred, reconst, mu, logvar, mu_prior, logvar_prior, qy_logit, qy

