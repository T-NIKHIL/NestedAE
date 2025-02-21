import pytest

# Complete set of supported options for nn_train_params_dict
@pytest.fixture
def sample_nn_train_params_dict():
    return {'global_seed':100, 
                # Required
                # Desc : Set the global random seed
                # Supported input type : int
            'epochs':1500,
                # Required
                # Desc : Number of epochs
                # Supported input type : int
            'batch_size':1,
                # Required
                # Desc : Batch size
                # Supported input type : int
            'val_split':0.1, 
                # Required
                # Desc : Fraction of data to use for validation
                # Supported input type : float
            'shuffle_data_between_epochs':True, 
                # Optional (default is True)
                # Desc : Fraction of data to use for validation
                # Supported input type : float
            'optimizer':{'type':'adam', 
                         'module_name':['encoder', 'predictor', 'decoder'], 
                         'lr':[1e-3, 1e-3, 1e-3]},
                # Required
                # Desc : Optimizer type and learning rate for each module
                # Supported input type : dict
            'callbacks':{'model_checkpoint':{'monitor':'total_val_loss', #Optional 
                                             'save_top_k':1,
                                             'mode':'min'}}
                # Optional
                # Desc : Callbacks for training
                # Supported input type : dict
            }

