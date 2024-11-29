""" Script that calls create_preprocessed_datasets() to preprocess the data."""

import sys
import json
import logging, os
logging.disable(logging.WARNING)

import click
from inputs.dataset_inputs import list_of_nn_datasets_dict
from inputs.nn_inputs import list_of_nn_params_dict
from inputs.train_inputs import list_of_nn_train_params_dict

"Hello"

def set_global_random_seed(seed):
    """Sets the global random seed."""

    # Pytorch lightning function to seed RNG for everything
    # Setting workers=True, Lightning derives unique seeds 
    # 1. across all dataloader workers and processes for torch
    # 2. numpy and stdlib random number generators
    pl.seed_everything(seed, workers=True)

    # Seeds RNG for all devices (CPU and GPU)
    torch.manual_seed(seed)

    # Sets the random seed for python
    random.seed(seed)

    # Sets the random seed in numpy library
    np.random.seed(seed)

@click.command()
@click.option('--run_dir', prompt='run_dir', 
              help='The run directory contains all the NestedAE models trained on the multiscale dataset.')
@click.option('--ae_save_dir', prompt='ae_save_dir', 
              help='Specify name of the directory to store the model.')
@click.option('--ae_idx', prompt='ae_idx',  
              help='Specify neural network number used for making the prediction.')
@click.option('--mode', prompt='mode', 
              help='Specify whether to preprocess train or predict data.')
@click.option('--plot_feats_dist', prompt='plot_feats_dist',
              help='Specify whether to plot the features distribution.')
@click.option('--num_cluster_lower', prompt='num_cluster_lower',
              help='Specify the lower limit of the number of clusters.')
@click.option('--num_cluster_upper', prompt='num_cluster_upper',
                help='Specify the upper limit of the number of clusters.')
@click.option('--cluster_metric', prompt='cluster_metric',
                help='Specify the metric to use for clustering.')
def preprocess_data(run_dir, ae_save_dir, ae_idx, mode, plot_feats_dist, num_cluster_lower, num_cluster_upper, cluster_metric):

    # Make the dir to contain all the runs 
    if not os.path.exists('../runs'):
        os.mkdir('../runs')
        print(' --> ../runs directory created.')
    else:
        print(' --> ../runs directory already exists.')

    # Make the run dir
    if not os.path.exists(f'../runs/{run_dir}'):
        os.mkdir(f'../runs/{run_dir}')
        print(f' --> ../runs/{run_dir} directory created.')
    else:
        print(f' --> ../runs/{run_dir} directory already exists.')

    ae_save_dir_path = f'../runs/{run_dir}/{ae_save_dir}'
    if os.path.exists(ae_save_dir_path) is False:
        os.mkdir(ae_save_dir_path)
        print(f' --> ../runs/{run_dir}/{ae_save_dir} directory created.')
    else:
        print(f' --> ../runs/{run_dir}/{ae_save_dir} directory already exists.')

    ae_idx = int(ae_idx)
    
    # Read in the input dictionaries


    nn_params_dict = list_of_nn_params_dict[ae_idx]
    nn_train_params_dict = list_of_nn_train_params_dict[ae_idx]
    nn_datasets_dict = list_of_nn_datasets_dict[ae_idx]

    if len(nn_params_dict.keys()) == 0:
        raise ValueError(' --> Provided empty dictionary nn params dictionary !')

    if len(nn_train_params_dict.keys()) == 0:
        raise ValueError(' --> Provided empty dictionary nn train params dictionary !')

    if len(nn_datasets_dict.keys()) == 0:
        raise ValueError(' --> Provided empty dictionary nn datasets dictionary !')

    # Send all print statements to file for debugging
    print_file_path = ae_save_dir_path + '/' + 'preprocess_data_out.txt'
    sys.stdout = open(print_file_path, "w", encoding='utf-8')

    print(f' --> User provided command line run_dir argument : {run_dir}')
    print(f' --> User provided command line nn argument : {ae_idx}')
    print(f' --> User provided command line mode argument : {mode}')

    ################################################################################################
    # Preprocess user provided nn dictionaries
    ################################################################################################

    # Check if required keys are present in nn_train_params_dict
    _required_keys = ['global_seed',
                      'epochs', 
                      'batch_size',
                      'shuffle_data_between_epochs',
                      'optimizer',
                      'test_split']

    _required_keys_dtypes = [int, int, int, bool, dict, float]

    _provided_keys = set(list(nn_train_params_dict.keys()))

    # Check if required keys are preset in the nn dictionary
    if _provided_keys.issuperset(set(_required_keys)) is False:
        missing_keys = set(_required_keys).difference(_provided_keys)
        raise KeyError(f' --> Missing {missing_keys} in nn train params dict.')

    # Typecast entry to required collection or data type
    for i, _required_key in enumerate(_required_keys):
        if isinstance(nn_train_params_dict[_required_key], _required_keys_dtypes[i]) is False:
            raise TypeError(f' --> Value for {_required_key} key in nn_train_params_dictionary should be of type {_required_keys_dtypes[i]}.')
        
    # Perform same check for nn_params_dict
    _required_submodule_keys = ['connect_to',
                                'output_dim',
                                'hidden_dim',
                                'hidden_layers',
                                'layer_type',
                                'layer_kernel_init',
                                'layer_bias_init']

    _required_loss_keys = ['type',
                           'wt',
                           'target']

    for submodule_name, submodule_dict in \
        zip(nn_params_dict['submodules'].keys(), nn_params_dict['submodules'].values()):

        # Make sure all the required keys are there in module dictionary
        submodule_keys = set(list(submodule_dict.keys()))

        # 'z' is the latent submodule
        if submodule_name != 'z':
            
            # Check if path is correct when loading submodule
            if 'load_submodule' in submodule_keys:
                path = submodule_dict['load_submodule']
                if os.path.exists(path) is False:
                    raise FileNotFoundError(f' --> Unable to find {path} to read submodule from.')
                else:
                    continue
            
            # Check if the submodule keys provided is a superset of the required keys
            if submodule_keys.issuperset(set(_required_submodule_keys)) is False:
                missing_keys = set(_required_submodule_keys).difference(submodule_keys)
                raise KeyError(f' --> Missing {missing_keys} in submodule \
                                {submodule_name} dictionary.')

            # If a submodule output is evaluated using a loss function, then associated loss keys should be present
            if 'loss' in submodule_keys:

                if isinstance(submodule_dict['loss'], dict) is False:
                    raise TypeError(' --> Value for "loss" key should be a dictionary.')

                loss_keys = set(list(submodule_dict['loss'].keys()))

                if loss_keys.issuperset(set(_required_loss_keys)) is False:
                    missing_keys = set(_required_loss_keys).difference(loss_keys)
                    raise KeyError(f' --> Missing {missing_keys} key in submodule \
                                    {submodule_name} loss dictionary.')
                
    # Check if paths to all directories provided in nn datasets dictionary exist
    datasets = nn_datasets_dict[mode]
    for dataset_dict in list(datasets.values()):
        path = dataset_dict['path']
        if os.path.exists(path) is False:
            raise FileNotFoundError(f' --> Unable to find {path} to read dataset from.')
    
    # Set the global random seed
    global_seed = nn_train_params_dict['global_seed']
    set_global_random_seed(global_seed)
    print(f' --> Set global random seed {global_seed}.')

    # Save the history of all different models created in the run directory.
    with open(ae_save_dir_path + '/' + 'run_summary.txt', 'a', encoding='utf-8') as file:
        file.write(f'--NN params dict (Model {ae_idx})--' + '\n')
        file.write(json.dumps(
            list_of_nn_params_dict[ae_idx], indent=4) + '\n')
        file.write('\n')

        file.write(
            f'--NN train params dict (Model {ae_idx})--' + '\n')
        file.write(json.dumps(
            list_of_nn_train_params_dict[ae_idx], indent=4) + '\n')
        file.write('\n')

        file.write(
            f'--NN dataset dict (Model {ae_idx})--' + '\n')
        file.write(json.dumps(
            list_of_nn_datasets_dict[ae_idx], indent=4) + '\n')
        file.write('\n')
    print(' --> Saved user provided dictionaries to run_summary.txt')
        
    test_split = nn_train_params_dict['test_split']
    # Used in predict mode, to create datasets for model inference
    dataset, variable_preprocessors = create_preprocessed_datasets(ae_save_dir_path, nn_datasets_dict, mode=mode)
    # For training mode, split the dataset into train and validation
    if mode == 'train':
        create_train_val_datasets(dataset, test_split, variable_preprocessors, 
                                  num_cluster_lower, num_cluster_upper, cluster_metric,
                                  global_seed=global_seed, 
                                  dataset_save_dir=f'../runs/{run_dir}/{ae_save_dir}/datasets',
                                  plot_feats_dist=plot_feats_dist)

    print(' --> Preprocessed dataset.')
    print(' --> PROGRAM EXIT.')

" Dataset utils script "

import os

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance

class CreateDataset(Dataset):
    """ Creates a PyTorch Dataset. """

    def __init__(self, name, dataset, variable_names,
                 variable_shapes, variable_preprocessors, variable_dtypes):

        # List of tensors
        self.name = name
        self.dataset = dataset

        self.variable_names = variable_names
        self.variable_shapes = variable_shapes
        self.variable_preprocessors = variable_preprocessors
        self.variable_dtypes = variable_dtypes

        # Get all the variable shapes
        shapes = list(self.variable_shapes.values())

        num_samples = shapes[0][0]

        sum_of_variables_shape = sum([shape[1] for shape in shapes])

        self.shape = (num_samples, sum_of_variables_shape)

    # Returns the number of samples in each dataset
    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        item = {}
        for variable_name in self.variable_names:
            item[variable_name] = self.dataset[variable_name][idx, :]

        return item


def load_csv_file(file, header, skiprows):
    """ Reads the csv file into a pandas dataframe

    Args:
        header : Row (0-indexed) to use as column labels for the dataframe
        index_col : Column (0-indexed) to use as the row labels for the datafram

    Returns : numpy array
    """
    dataframe = pd.read_csv(file, header=header, skiprows=skiprows)
    return dataframe


def load_xlsx_file(file, sheet_name, header, skiprows):
    """ Reads the excel file into a pandas dataframe

    Args:
        sheet_name : Name of the excel sheet to read.
        header : Row (0-indexed) to use as column labels for the dataframe
        index_col : Column (0-indexed) to use as the row labels for the datafram

    Returns : numpy array
    """
    dataframe = pd.read_excel(file, sheet_name=sheet_name,
                              header=header, skiprows=skiprows)
    return dataframe


def load_npy_file(file):
    """ Loads the .npy, .npz or pickled files into a numpy array"""
    return np.load(file)


def one_hot_encode(samples, variable):
    """ One hot encodes the samples for the variable"""
    _ohe = OneHotEncoder(dtype=np.float32)
    _ohe.fit(samples)
    samples_ohe = _ohe.transform(samples).toarray()

    categories = _ohe.categories_[0]

    dtype = samples_ohe.dtype
    shape = samples_ohe.shape

    print(f' --> {variable} dtype : {dtype}')
    print(f' --> {variable} dim   : {shape}')

    return samples_ohe, _ohe, dtype, shape, categories


def label_binarizer(samples, variable):
    """ Label binarizes the samples for the variable"""
    _lb = LabelBinarizer(dtype=np.float32)
    _lb.fit(samples)
    samples_lb = _lb.transform(samples)

    dtype = samples_lb.dtype
    shape = samples_lb.shape

    print(f' --> {variable} dtype : {dtype}')
    print(f' --> {variable} shape : {shape}')

    return samples_lb, _lb, dtype, shape


def label_encoder(samples, variable):
    """Label encodes the samples for the variable"""
    _le = LabelEncoder()
    _le.fit(samples)
    samples_le = _le.transform(samples)

    dtype = samples_le.dtype
    shape = samples_le.shape

    print(f' --> {variable} dtype : {dtype}')
    print(f' --> {variable} shape : {shape}')

    return samples_le, _le, dtype, shape


def standardize(samples, variable):
    """ Standardizes the samples for the variable"""
    samples = samples.astype(np.float32)
    _ss = StandardScaler()
    samples_std = _ss.fit_transform(samples)

    dtype = samples_std.dtype
    shape = samples_std.shape

    print(f' --> {variable} dtype : {dtype}')
    print(f' --> {variable} shape : {shape}')

    return samples_std, _ss, dtype, shape


def create_preprocessed_datasets(nn_save_dir, nn_dataset_dict, mode='train'):
    """ Reads data from specified source and processes it according to specifications in nn_dataset_dict

    Args:
        nn_save_dir: Model directory (Required)
        nn_dataset_dict: The dictionary containing all 
        preprocessing specifications for the dataset (Required)
        global_seed: The seed used for shuffling (Required)

    Returns: None
    """
    datasets = nn_dataset_dict[mode]

    dataset_dicts = list(datasets.values())
    dataset_names = list(datasets.keys())

    # Create a directory to store all the datasets
    # Check if run directory already exists
    print(nn_save_dir)
    dataset_save_dir = nn_save_dir + '/datasets'
    if os.path.exists(dataset_save_dir) is False:
        os.mkdir(dataset_save_dir)
        print(' --> Dataset directory created.')
    else:
        print(' --> Dataset directory already exists. Proceeding to rewrite.')

    # Create one dataset that combines data from all datasets
    dataset = {}
    tensor_dataset = {}
    variable_preprocessors = {}
    variable_dtypes = {}
    variable_shapes = {}
    all_variable_names = []
    sample_wts = None

    # Outer loop iterates over each dataset (Ex : X, y, latents ...)
    for i, dataset_dict in enumerate(dataset_dicts):

        dataset_name = dataset_names[i]

        # nn_name = nn_save_dir.split('/')[-1]

        dataset_file_path = dataset_dict['path']
        dataset_file_name = dataset_file_path.split('/')[-1]
        dataset_file_type = dataset_file_name.split('.')[-1]

        if dataset_file_type == 'npy':
            print(f' --> Found .npy file : {dataset_file_name}')
            dataframe = load_npy_file(dataset_file_path)
        elif dataset_file_type == 'xlsx':
            print(f' --> Found .xlsx file : {dataset_file_name}')
            try:
                sheet_name = dataset_dict['sheet_name']
                header = dataset_dict['header']
                skiprows = dataset_dict['skiprows']
            except:
                sheet_name = 0
                header = 0
                skiprows = None
            dataframe = load_xlsx_file(dataset_file_path, sheet_name,
                                       header, skiprows)
        elif dataset_file_type == 'csv':
            print(f' --> Found .csv file : {dataset_file_name}')
            try:
                header = dataset_dict['header']
                skiprows = dataset_dict['skiprows']
            except:
                header = 0
                skiprows = None
            dataframe = load_csv_file(dataset_file_path,
                                      header, skiprows)
        else:
            raise FileNotFoundError(' --> Supported file type not found')

        print(f' --> Loaded {dataset_name} as a dataframe.')
        print(f' --> {dataset_name} Dataframe shape : {dataframe.shape}')

        print(' --> Dataframe head.')
        print(dataframe.head())

        # data_id is string identifier for the pandas df col / df cols
        variable_dicts = list(dataset_dict['variables'].values())
        variable_names = list(dataset_dict['variables'].keys())

        if i == 0:
            all_variable_names = variable_names
        else:
            all_variable_names.extend(variable_names)

        # Inner loop iterates over each variable in the dataset
        for j, variable_dict in enumerate(variable_dicts):

            variable_name = variable_names[j]

            cols = variable_dict['cols']
            cols = cols if not isinstance(cols, list) else list(cols)

            preprocess_scheme = variable_dict['preprocess']

            # Check for any NA values in variable
            na_values = dataframe.iloc[:, cols].isna()

            if na_values.any().any():
                raise ValueError(f' --> NA values found in {variable_name} \
                        dataframe. Check log file for details.')
            else:
                print(f' --> No NA values found in {variable_name}.')

            # Extract samples for variable
            samples = dataframe.iloc[:, cols].values

            if samples.shape[1] == 1:
                samples = samples.reshape(-1, 1)

            print(
                f' --> Extracting data for {variable_name} from {dataset_name} dataframe cols {cols}.')

            # Data Preprocessing section

            if preprocess_scheme is None:
                samples = samples.astype(np.float32)
                print(f' --> No preprocessing done for {variable_name} \
                            from {dataset_name} dataframe cols {cols}.')
                preprocessor = None
                dtype = samples.dtype
                shape = samples.shape
                print(f' --> {variable_name} dtype : {dtype}')
                print(f' --> {variable_name} dim   : {shape}')
            elif preprocess_scheme == 'std':
                if dataset_dict['load_preprocessor'] is True:
                    print(f' --> Loading scaler for {variable_name}.')
                    loaded_dataset = torch.load(
                        dataset_save_dir + '/dataset.pt')
                    preprocessor = loaded_dataset.variable_preprocessors[variable_name]
                    dtype = loaded_dataset.variable_dtypes[variable_name]
                    shape = loaded_dataset.variable_shapes[variable_name]
                    samples = preprocessor.transform(samples)
                else:
                    samples, preprocessor, dtype, shape = standardize(
                        samples, variable_name)
            elif preprocess_scheme == 'ohe':
                if dataset_dict['load_preprocessor'] is True:
                    print(f' --> Loading one hot encoder for {variable_name}.')
                    loaded_dataset = torch.load(
                        dataset_save_dir + '/dataset.pt')
                    preprocessor = loaded_dataset.variable_preprocessors[variable_name]
                    dtype = loaded_dataset.variable_dtypes[variable_name]
                    shape = loaded_dataset.variable_shapes[variable_name]
                    samples = preprocessor.transform(samples)
                else:
                    samples, preprocessor, dtype, shape, categories = one_hot_encode(
                        samples, variable_name)
                    print(f' --> Encoded col {cols} \
                        with {len(categories)} \
                        categories {categories}')
            elif preprocess_scheme == 'lb':
                samples, preprocessor, dtype, shape = label_binarizer(
                    samples, variable_name)
            elif preprocess_scheme == 'le':
                samples, preprocessor, dtype, shape = label_encoder(
                    samples, variable_name)
            else:
                raise ValueError(' --> Preprocessing scheme not defined. ')

            if i == 0 and j == 0:
                samples_preview = samples
            else:
                samples_preview = np.hstack((samples_preview, samples))
            dataset[variable_name] = samples
            tensor_dataset[variable_name] = torch.tensor(
                samples, dtype=torch.float32)
            variable_preprocessors[variable_name] = preprocessor
            variable_dtypes[variable_name] = dtype
            variable_shapes[variable_name] = shape

        if 'weight_samples' in list(dataset_dict.keys()) is True:
            sample_wts = weight_samples(dataframe, dataset_dict)
            all_variable_names.append('sample_wts')
            # Add the sample wts to the preview
            samples_preview = np.hstack((samples_preview, sample_wts))
            tensor_dataset['sample_wts'] = torch.tensor(
                sample_wts, dtype=torch.float32)
            variable_preprocessors['sample_wts'] = None
            variable_dtypes['sample_wts'] = sample_wts.dtype
            variable_shapes['sample_wts'] = sample_wts.shape
            print(f' --> sample_wts dtype : {sample_wts.dtype}')
            print(f' --> sample_wts shape : {sample_wts.shape}')
    if mode == 'train':
        # Save the processed data to .csv file for easy preview
        preview_file_name = f'{dataset_name}_dataset_preview.csv'
        np.savetxt(dataset_save_dir + '/' + preview_file_name,
                   samples_preview, delimiter=',')
        ae_dataset = CreateDataset(name=dataset_name,
                                   dataset=tensor_dataset,
                                   variable_names=all_variable_names,
                                   variable_shapes=variable_shapes,
                                   variable_preprocessors=variable_preprocessors,
                                   variable_dtypes=variable_dtypes)
        pickle_file_name = f'{dataset_name}_dataset.pt'
        torch.save(ae_dataset, dataset_save_dir + '/' + pickle_file_name)
    elif mode == 'predict':
        # Save the processed data to .csv file for easy preview
        preview_file_name = f'{dataset_name}_dataset_preview.csv'
        np.savetxt(dataset_save_dir + '/' + preview_file_name,
                   samples_preview, delimiter=',')
        pickle_file_name = f'{dataset_name}_dataset.pt'
        torch.save(tensor_dataset, dataset_save_dir + '/' + pickle_file_name)
    else:
        raise ValueError(f'Invalid mode specified {mode}.')
    return dataset, variable_preprocessors

def create_kfold_datasets(dataset, n_splits, variable_preprocessors, global_seed=0, dataset_name=None, dataset_save_dir='.'):
    """ Creates kfold datasets from the dataset provided."""
    # Create a cross validator
    cross_validator = KFold(
        n_splits=n_splits, shuffle=True, random_state=global_seed)
    # Get the first variable out of the dataset
    variable_names = list(dataset.keys())
    first_variable_samples = dataset[variable_names[0]]
    idxs = cross_validator.split(first_variable_samples)
    # Iterate over the folds
    for i, (train_idx, test_idx) in enumerate(idxs):
        print(
            f' -->Fold {i} : train_idx {train_idx.shape} test_idx {test_idx.shape}')
        print(f' -->Train idxs : {train_idx}')
        print(f' -->Test idxs : {test_idx}')
        train_tensor_dataset = {}
        val_tensor_dataset = {}
        train_var_shapes = {}
        val_var_shapes = {}
        train_var_dtypes = {}
        val_var_dtypes = {}
        # Iterate over the variables in the dataset
        for j, variable_name in enumerate(variable_names):
            train_samples = dataset[variable_name][train_idx][:]
            val_samples = dataset[variable_name][test_idx][:]
            train_tensor_dataset[variable_name] = torch.tensor(
                train_samples, dtype=torch.float32)
            val_tensor_dataset[variable_name] = torch.tensor(
                val_samples, dtype=torch.float32)
            if j == 0:
                train_samples_preview = train_samples
                val_samples_preview = val_samples
            else:
                train_samples_preview = np.hstack(
                    (train_samples_preview, train_samples))
                val_samples_preview = np.hstack(
                    (val_samples_preview, val_samples))
            train_var_shapes[variable_name] = train_samples.shape
            val_var_shapes[variable_name] = val_samples.shape
            train_var_dtypes[variable_name] = train_samples.dtype
            val_var_dtypes[variable_name] = val_samples.dtype
        # Store the datasets fro each fold and dataset previews
        ae_train_dataset = CreateDataset(name=dataset_name,
                                         dataset=train_tensor_dataset,
                                         variable_names=variable_names,
                                         variable_preprocessors=variable_preprocessors,
                                         variable_dtypes=train_var_dtypes,
                                         variable_shapes=train_var_shapes)
        pickle_file_name = f'train_dataset_fold_{i}.pt'
        torch.save(ae_train_dataset, dataset_save_dir + '/' + pickle_file_name)
        ae_val_dataset = CreateDataset(name=dataset_name,
                                       dataset=val_tensor_dataset,
                                       variable_names=variable_names,
                                       variable_preprocessors=variable_preprocessors,
                                       variable_dtypes=val_var_dtypes,
                                       variable_shapes=val_var_shapes)
        pickle_file_name = f'val_dataset_fold_{i}.pt'
        torch.save(ae_val_dataset, dataset_save_dir + '/' + pickle_file_name)
        print(' --> Saved dataset to pickle under /datasets directory.')
        np.savetxt(dataset_save_dir + '/' + f'train_dataset_preview_fold_{i}.csv',
                   train_samples_preview, delimiter=',', header=','.join(variable_names))
        np.savetxt(dataset_save_dir + '/' + f'val_dataset_preview_fold_{i}.csv',
                   val_samples_preview, delimiter=',', header=','.join(variable_names))
        print(f' --> Created {i} fold for dataset.')
        print(f' --> Number of variables in dataset {len(dataset)}.')
        print(f' --> Train dataset shape : {train_var_shapes}.')
        print(f' --> Val dataset shape :{val_var_shapes}.')

def create_train_val_datasets(dataset, test_split, variable_preprocessors, num_cluster_lower, num_cluster_upper, cluster_metric, global_seed=0, dataset_save_dir='.', plot_feats_dist=False):
    """ Creates a train and test list consisting of the train and test numpy arrays respectively."""

    # Step 1 : Dimensionality reduction on the dataset using PCA for visualization
    feats_list = [dataset[key] for key in dataset.keys()]
    for i, feat_list in enumerate(feats_list):
        if i == 0:
            numpy_dataset = feat_list
        else:
            numpy_dataset = np.hstack((numpy_dataset, feat_list))
    pca = PCA(n_components=2, random_state=global_seed)
    pca_coords = pca.fit_transform(numpy_dataset)
    plt.scatter(pca_coords[:, 0], pca_coords[:, 1])
    plt.savefig(dataset_save_dir + '/2d_pca.png')

    # Step 2 : KMeans on the original dataset
    kmeans_param_grid = {"n_clusters":range(int(num_cluster_lower), int(num_cluster_upper))}

    def scoring(metric):
        if metric == 'silhouette':
            def kmeans_silhouette_score(estimator, X):
                y_pred = estimator.predict(X)
                return silhouette_score(X, y_pred)
            return kmeans_silhouette_score
        elif metric == 'calinski_harabasz':
            def kmeans_calinski_harabasz_score(estimator, X):
                y_pred = estimator.predict(X)
                return calinski_harabasz_score(X, y_pred)
            return kmeans_calinski_harabasz_score
        elif metric == 'davies_bouldin':
            def kmeans_davies_bouldin_score(estimator, X):
                y_pred = estimator.predict(X)
                return davies_bouldin_score(X, y_pred)
            return kmeans_davies_bouldin_score
        else:
            raise ValueError(f' --> Scoring metric {metric} not found.')

    grid_search_kmeans = GridSearchCV(
        KMeans(n_init=10, max_iter=300, random_state=global_seed), 
        param_grid=kmeans_param_grid, 
        scoring=scoring(cluster_metric))
    grid_search_kmeans.fit(numpy_dataset)
    df_grid_search_kmeans_results = pd.DataFrame(grid_search_kmeans.cv_results_)[
        ["param_n_clusters", "mean_test_score"]
    ]
    print(' --> KMeans clustering results')
    print(df_grid_search_kmeans_results.sort_values(by="mean_test_score", ascending=False))

    # Step 3 : Predict the clusters each PCA coord belongs to for the best kmeans model
    cluster_preds = grid_search_kmeans.best_estimator_.predict(numpy_dataset)
    # cluster_preds = KMeans(n_clusters=186).fit_predict(numpy_dataset)
    plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c=cluster_preds)
    plt.savefig(dataset_save_dir + '/2d_pca_clustered.png')
    plt.close()

    # Step 4 : Splitting the data in each cluster into train and test based on the test_split
    x_train = []
    x_val = []
    for cluster in np.unique(cluster_preds):
        cluster_samples = numpy_dataset[cluster_preds == cluster]
        if len(cluster_samples) == 1:
            x_train.append(cluster_samples)
            continue
        x_train_cluster, x_val_cluster = train_test_split(cluster_samples, test_size=test_split, random_state=global_seed)
        x_train.append(x_train_cluster)
        x_val.append(x_val_cluster)
    x_train = np.concatenate(x_train)
    x_val = np.concatenate(x_val)

    # Step 5 : Get the train and val feature distribution
    # Convert X_train_inv to a dataframe and plot histograms of the features
    if plot_feats_dist:
        x_variables = ['A_ion_radius', 'A_atomic_wt', 'A_EA', 'A_IE', 'A_EN', 'B_ion_radius', 'B_atomic_wt', 'B_EA', 'B_IE', 'B_EN', 'X_ion_radius', 'X_atomic_wt', 'X_EA', 'X_IE', 'X_EN', 'bg']
        x_train_df = pd.DataFrame(x_train, columns=x_variables)
        x_val_df = pd.DataFrame(x_val, columns=x_variables)

        nbins=10
        _, axs = plt.subplots(4, 5, figsize=(22, 12))

        row_ax = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3]
        col_ax = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]

        for i, feature in enumerate(x_variables):
            # Plotting the histogram
            sns.histplot(x_train_df[feature], bins=nbins, ax=axs[row_ax[i], col_ax[i]], color='blue', label='Train', kde=True, kde_kws={'bw_method':'scott'})
            sns.histplot(x_val_df[feature], bins=nbins, ax=axs[row_ax[i], col_ax[i]], color='red', label='Test', kde=True, kde_kws={'bw_method':'scott'})
            train_kde_samples = x_train_df[feature].sample(10000, random_state=global_seed, replace=True)
            test_kde_samples = x_val_df[feature].sample(10000, random_state=global_seed, replace=True)
            # train_kde_samples = gaussian_kde(X_train_inv_df[feature]).resample(10000).reshape(-1)
            # test_kde_samples = gaussian_kde(X_test_inv_df[feature]).resample(10000).reshape(-1)
            axs[i//5, i%5].set_title(feature + f' KS pval:{round(ks_2samp(x_train_df[feature], x_val_df[feature]).pvalue,4)} WD :{round(wasserstein_distance(train_kde_samples, test_kde_samples),4)}')
        plt.tight_layout()    
        plt.savefig(dataset_save_dir + '/train_val_feature_distributions.png')      
        plt.close()

    # Step 6 : Create PyTorch datasets
    train_tensor_dataset = {}
    val_tensor_dataset = {}
    train_var_shapes = {}
    val_var_shapes = {}
    train_var_dtypes = {}
    val_var_dtypes = {}
    for variable_name in dataset.keys():
        # Get the number of columns
        varaible_shape = dataset[variable_name].shape[1]
        train_tensor_dataset.update({variable_name: torch.tensor(x_train[:, :varaible_shape], dtype=torch.float32)})
        val_tensor_dataset.update({variable_name: torch.tensor(x_val[:, :varaible_shape], dtype=torch.float32)})
        train_var_shapes[variable_name] = train_tensor_dataset[variable_name].shape
        val_var_shapes[variable_name] = val_tensor_dataset[variable_name].shape
        train_var_dtypes[variable_name] = train_tensor_dataset[variable_name].dtype
        val_var_dtypes[variable_name] = val_tensor_dataset[variable_name].dtype
    
    ae_train_dataset = CreateDataset(name='train',
                                        dataset=train_tensor_dataset,
                                        variable_names=list(dataset.keys()),
                                        variable_preprocessors=variable_preprocessors,
                                        variable_dtypes=train_var_dtypes,
                                        variable_shapes=train_var_shapes)
    pickle_file_name = 'train_dataset.pt'
    torch.save(ae_train_dataset, dataset_save_dir +
                '/' + pickle_file_name)
    ae_val_dataset = CreateDataset(name='val',
                                    dataset=val_tensor_dataset,
                                    variable_names=list(dataset.keys()),
                                    variable_preprocessors=variable_preprocessors,
                                    variable_dtypes=val_var_dtypes,
                                    variable_shapes=val_var_shapes)
    pickle_file_name = 'val_dataset.pt'
    torch.save(ae_val_dataset, dataset_save_dir +
                '/' + pickle_file_name)
    print(' --> Saved dataset to pickle under /datasets directory.')
    # Save the train and val samples preview to a csv
    preview_file_name = 'train_dataset_preview.csv'
    np.savetxt(dataset_save_dir + '/' + preview_file_name, x_train, delimiter=',')
    preview_file_name = 'val_dataset_preview.csv'
    np.savetxt(dataset_save_dir + '/' + preview_file_name, x_val, delimiter=',')
    print(' --> Created train and val datasets for dataset.')
    print(f' --> Number of variables in dataset {len(dataset)}.')
    print(f' --> Train dataset shape : {train_var_shapes}.')
    print(f' --> Val dataset shape :{val_var_shapes}.')

def weight_samples(dataframe, dataset_dict):
    """ Weight samples based on provided weighting scheme."""

    col_idx = dataset_dict['weight_samples']['col_idx']
    nbins = dataset_dict['weight_samples']['nbins']
    scheme = dataset_dict['weight_samples']['scheme']

    # First check the data type of the dataframe column
    samples = dataframe.iloc[:, col_idx].values

    if is_numeric_dtype(samples):
        print(' --> Dataframe column is of numeric data type')
        print(
            f' --> Creating a histogram for the numeric sample with {nbins} bins')

        hist, bin_edges = np.histogram(samples, bins=nbins)

        # Righ most bin is hlaf open
        bin_num_for_each_sample = np.digitize(samples, bin_edges, right=False)

        # Get the counts for each bin_num
        bin_nums, bin_num_counts = np.unique(
            bin_num_for_each_sample, return_counts=True)

        if scheme == 'inv_count':
            wts_for_each_bin = 1/bin_num_counts
        else:
            raise ValueError(
                f' --> Requested weighting scheme {scheme} is not available.')

        # Assign the sample weights to each sample
        sample_wts = []
        for bin_num in bin_num_for_each_sample:
            sample_wts.append(wts_for_each_bin[bin_num-1])

        sample_wts = np.array(sample_wts, dtype=np.float32).reshape(-1, 1)

        # Create a new column in the dataframe to store the sample
        # dataframe['sample_wts'] = sample_wts

        return sample_wts

    elif is_string_dtype(samples):
        print(' --> Dataframe column is of string data type.')

        # Get the different classes for the categorical variable and the counts for each class
        value_counts = dataframe.iloc[:, df_col].value_counts()

        counts = value_counts.values
        classes = [cl for cl in value_counts.index.values.tolist()]

        for i, cl in enumerate(classes):
            print(f' --> Found the {cl} class with {counts[i]} counts.')

        # Assign sample weighting to each sample in the class
        print(f' --> Applying the {scheme} weight scheme.')
        sample_weights = []

        # Iterate through the samples
        for sample in samples:
            # Find which class the sample belongs to
            j = classes.index(sample)
            sample_weights.append(generate_sample_weight(scheme, counts[j]))

        return sample_wts

    else:
        raise ValueError(
            'Dataframe contains a data type different from numeric and string.')



if __name__ == '__main__':
    preprocess_data()
