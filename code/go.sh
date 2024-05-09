#!/bin/bash

## source ~/miniconda/bin/activate

# <-- USER START -->

## Specify folder to look at for the input files
inputs_dir=inputs_perov_data_v2
## Specify folder to save the NestedAE outputs to
run_dir=perovskite_multiscale_dataset_v2
## Which neural network to use for training or inference
nn=1
## Specify mode of operation. train or predict (Ignored when tuning neural network parameters)
mode=train
## Specify accelerator to use. cpu or gpu
accelerator=cpu
## Name of submodule from which predictions are required.
## Use when selecting 'predict' mode. Ignored in 'train' mode (Ignored when tuning neural network parameters)
submodule=bg_pred
## Number of kfold datasets to create to test the robustness of model parameters
kfolds=5

## Tuning neural network parameters using wandb
tune_nn_params=true
user_name=nthota2
project_name=$run_dir
sweep_type=grid
metric=total_val_loss
goal=minimize
trials_in_sweep=-1 # -1 for all trials

# <-- USER END -->

if [[ ! -d ../runs ]]; then
    mkdir ../runs
fi

## The inputs directory can be any name.
## The code copies inputs_dir to inputs.
rm -rf inputs
cp -rf $inputs_dir inputs

if [[ $tune_nn_params == true ]]; then
	python3 tune_nn_params.py --run_dir $run_dir --nn $nn --kfolds $kfolds --user_name $user_name --project_name $project_name --sweep_type $sweep_type --metric $metric --goal $goal --trials_in_sweep $trials_in_sweep --accelerator $accelerator
	wait
	rm -rf inputs
else
	## Uncomment '##' for to skip preprocessing. By default, this section of code will run preprocessing of data for NestedAE
	##<<comment
	python3 preprocess_data.py --run_dir $run_dir --nn $nn --mode $mode --kfolds $kfolds &
	wait
	rm -rf inputs
	##comment
	if [[ $mode == train ]]; then
		echo "In train mode"
		python3 train.py --run_dir $run_dir --nn $nn --accelerator $accelerator
	fi 
	if [[ $mode == predict ]]; then
		echo "In predict mode"
		python3 predict.py --run_dir $run_dir --nn $nn --accelerator $accelerator --submodule $submodule
	fi
fi





