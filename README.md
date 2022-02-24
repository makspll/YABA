# mlp-cw4




## Set up environment
    - `conda create --name mlp-cw4`
	- `conda activate mlp-cw4`
    - `make installEnv` 


conda activate /conda_env
git pull
screen
python src/train.py --config configs/experiment_configs/frankle_resnet_56_cifar10.yaml
python src/graph.py --experiment_name sparse_resnet_56_cifar10 --graph_type all
rclone copy --drive-impersonate larsthalianmorstad@gmail.com FOLDER mlp-cw4:FOLDER
git add --all
git push