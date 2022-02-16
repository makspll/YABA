from csv import DictReader
from typing import Dict, List
import numpy as np
import torch 
import matplotlib.pyplot as plt
import logging 

logger = logging.getLogger(__name__)

def plot_bn_vs_other_gradient_magnitudes(gradient_epoch_filenames : List[str], training_stats_csv : str):

    # can't fit all in memory, so go through each filename
    bn_grad_average_mags = []
    non_bn_grads_average_mags = []

    val_accs = []
    train_accs = []

    with open(training_stats_csv,'r') as f:
        r = DictReader(f,fieldnames=["train_acc","train_loss","val_acc","val_loss"])
        next(r) # skip headers
        for row in r:
            val_accs.append(float(row['val_acc']))
            train_accs.append(float(row['train_acc']))


    # collect absolute magnitudes for bn vs non bn layers
    for checkpoint in gradient_epoch_filenames:
        logging.info(f"Processing: {checkpoint}")

        gradients = torch.load(checkpoint,map_location='cpu')['gradients']
        
        bn_mags = []
        non_bn_mags = []

        for param_name,gradient in gradients.items():
            if 'bias' in param_name: # no biases, or optimizer parameters
                continue

            abs_mean = gradient.abs().mean()

            if 'bn_' in param_name:
                bn_mags.append(abs_mean)
            else:
                non_bn_mags.append(abs_mean)

        # calculate averages 
        bn_grad_average_mags.append(np.mean(bn_mags))
        non_bn_grads_average_mags.append(np.mean(non_bn_mags))

    fig,ax_left = plt.subplots()
    ax_right = ax_left.twinx()



    ax_right.plot(range(1,len(gradient_epoch_filenames)+1),
                val_accs,label="val acc.",
                color='tab:red')
    ax_right.plot(range(1,len(gradient_epoch_filenames)+1),
                train_accs,label="train acc.",
                color='tab:orange')

    ax_left.plot(range(1,len(gradient_epoch_filenames)+1),
                bn_grad_average_mags,
                    label="BN",
                    color="tab:purple")

    ax_left.plot(range(1,len(gradient_epoch_filenames)+1),
                non_bn_grads_average_mags,
                    label="non-BN",
                    color='tab:pink')
    ax_left.legend()
    ax_right.legend()

    plt.title('BN vs non-BN layer average gradient magnitudes')
    plt.xlabel("Epoch")
    plt.ylabel("Average gradient magnitude (no biases)")

def plot_gradient_flow(gradient_epoch_filenames : List[str]):

    # can't fit all in memory, so go through each filename
    bn_grad_average_mags = []
    non_bn_grads_average_mags = []

    # collect absolute magnitudes for bn vs non bn layers
    for checkpoint in gradient_epoch_filenames:
        logging.info(f"Processing: {checkpoint}")

        params = []
        gradients = [] 
        for param_name,gradient in  torch.load(checkpoint,map_location='cpu')['gradients'].items():
            if 'bias' in param_name: # no biases, or optimizer parameters
                continue

            abs_mean = gradient.abs().mean()
            params.append(param_name)
            gradients.append(abs_mean)

        plt.plot(params,
                    gradients, color='blue')

    plt.xticks(range(0,len(params), 1), params, rotation="vertical")
    plt.tight_layout()
    plt.legend()
    plt.title('Gradient Flow')
    plt.xlabel("layer idx")
    plt.ylabel("Average gradient magnitude (no biases)")
