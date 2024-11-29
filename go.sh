#!/bin/bash

## source ~/miniconda/bin/activate

# <-- USER START -->

## Specify folder to look at for the input files
inputs_dir=inputs_perov_data
## Specify folder to save the NestedAE outputs to
run_dir=perovskite_multiscale_dataset_v3
ae_save_dir=ae1
## Which neural network to use for training or inference
ae_idx=0
## Specify mode of operation. preprocess, train or predict 
mode=preprocess
## Specify accelerator to use. cpu or gpu
accelerator=cpu
## Name of submodule from which predictions are required.
## Use when selecting 'predict' mode. Ignored in 'train' mode
submodule=bg_pred
plot_feats_dist=true
## 'k' range for k-means
num_cluster_lower=2
num_cluster_upper=50
## If using davies_bouldin, manually set num of clusters
cluster_metric=calinski_harabasz

## Training using wandb
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

if [[ $mode == preprocess ]]; then
	echo "In preprocess mode"
	python3 preprocess.py --run_dir $run_dir --ae_save_dir $ae_save_dir --ae_idx $ae_idx --mode train --plot_feats_dist $plot_feats_dist --num_cluster_lower $num_cluster_lower --num_cluster_upper $num_cluster_upper --cluster_metric $cluster_metric &
	wait
	rm -rf inputs
fi

if [[ $mode == train ]]; then
	echo "In train mode"
	python3 train.py --run_dir $run_dir --ae_save_dir $ae_save_dir --ae_idx $ae_idx --user_name $user_name --project_name $project_name --sweep_type $sweep_type --metric $metric --goal $goal --trials_in_sweep $trials_in_sweep --accelerator $accelerator
fi 

if [[ $mode == predict ]]; then
	echo "In predict mode"
	python3 predict.py --run_dir $run_dir --ae_save_dir $ae_save_dir --ae_idx $ae_idx --accelerator $accelerator --submodule $submodule
fi





