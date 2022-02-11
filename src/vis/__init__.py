from typing import Dict, List
import numpy as np
import torch 
import matplotlib.pyplot as plt
import logging 

logger = logging.getLogger(__name__)

def plot_bn_vs_other_gradient_magnitudes(gradient_epoch_filenames : List[str]):

    # can't fit all in memory, so go through each filename
    bn_grad_average_mags = []
    non_bn_grads_average_mags = []

    # collect absolute magnitudes for bn vs non bn layers
    for checkpoint in gradient_epoch_filenames:
        logging.info(f"Processing: {checkpoint}")

        gradients = torch.load(checkpoint,map_location='cpu')['gradients']

        bn_mags = []
        non_bn_mags = []

        for param_name,gradient in gradients.items():
            if 'weight' in param_name: # no biases, or optimizer parameters
                continue

            abs_mean = gradient.abs().mean()

            if 'bn_' in param_name:
                bn_mags.append(abs_mean)
            else:
                non_bn_mags.append(abs_mean)

        # calculate averages 
        bn_grad_average_mags.append(np.mean(bn_mags))
        non_bn_grads_average_mags.append(np.mean(non_bn_mags))

    plt.plot(range(1,len(gradient_epoch_filenames)+1),
                bn_grad_average_mags,
                    label="BN")

    plt.plot(range(1,len(gradient_epoch_filenames)+1),
                non_bn_grads_average_mags,
                    label="non-BN")
    plt.legend()
    plt.title('BN vs non-BN layer average gradient magnitudes')
    plt.xlabel("Epoch")
    plt.ylabel("Average gradient magnitude (no biases)")
