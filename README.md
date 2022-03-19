# Yet Another Batch-norm Paper
An investigation into the representational power of Batch Normalisation


## Framework
This repository contains a useful network trainer as well as all the experiment config files needed to run all experiements discussed in the paper 

## Set up environment
    - `conda create --name mlp-cw4`
	- `conda activate mlp-cw4`
    - `make installEnv` 


## Running experiments + Generating Graphs

```
conda activate /conda_env \
python src/train.py --config configs/experiment_configs/sparse_resnet_56_cifar10.yaml \
python src/graph.py --experiment_name sparse_resnet_56_cifar10 --graph_type all \
```
