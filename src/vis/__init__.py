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

            if gradient is None:
                abs_mean = 0
            else:
                abs_mean = gradient.abs().mean()


            if 'bn' in param_name:
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
    plt.ylabel("Average gradient magnitude")


def plot_bn_vs_other_gradient_magnitudes_relative(gradient_epoch_filenames : List[str]):

    # can't fit all in memory, so go through each filename
    bn_grad_average_mags = []
    non_bn_grads_average_mags = []

    # collect absolute magnitudes for bn vs non bn layers
    for checkpoint in gradient_epoch_filenames:
        logging.info(f"Processing: {checkpoint}")

        chkp = torch.load(checkpoint,map_location='cpu')
        params = chkp['model_state']
        gradients = chkp['gradients']

        bn_mags = []
        non_bn_mags = []

        for param_name,gradient in gradients.items():

            param = params[param_name]

            if gradient is None:
                abs_mean = 0
            else:
                abs_mean = (gradient / param).abs().mean()


            if 'bn' in param_name:
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
    plt.title('BN vs non-BN layer average relative gradient magnitudes')
    plt.xlabel("Epoch")
    plt.ylabel("Average relative gradient magnitude")


def plot_bn_vs_other_parameter_magnitudes(gradient_epoch_filenames : List[str]):

    # can't fit all in memory, so go through each filename
    bn_grad_average_mags = []
    non_bn_grads_average_mags = []

    # collect absolute magnitudes for bn vs non bn layers
    for checkpoint in gradient_epoch_filenames:
        logging.info(f"Processing: {checkpoint}")

        chkp = torch.load(checkpoint,map_location='cpu')
        params = chkp['model_state']

        bn_mags = []
        non_bn_mags = []

        for param_name,parameter in params.items():

# bn1.running_mean
# tensor(0.4746)
# bn1.running_var
            if parameter is None or "running_mean" in param_name or "running_var" in param_name:
                abs_mean = 0
            else:
                try:
                    abs_mean = parameter.abs().mean()
                except:
                    continue

            if 'bn' in param_name:
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
    plt.title('BN vs non-BN layer average parameter magnitudes')
    plt.xlabel("Epoch")
    plt.ylabel("Average parameter magnitude")

def plot_accuracy(train_accs, val_accs):
    


    plt.plot(range(1,len(train_accs)+1),
                train_accs,
                    label="train_acc")

    plt.plot(range(1,len(val_accs)+1),
                val_accs,
                    label="val_acc")
    plt.legend()
    plt.title('Training and Validation Accuracies')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

def plot_final_accuracy(test_accs, exp_types, title):
    
    depths = ["56", "110", "218"]
    print(test_accs)
    for i in range(len(exp_types)):
        if "moresparse" not in exp_types[i]:
            color = "blue"
            if "nobatchdecay_" in exp_types[i]:
                color = "orange"
            elif "noregulardecay" in exp_types[i]:
                color = "cyan"
            elif "resnet" in exp_types[i]:
                color = "green"
                if "nodecay_" in exp_types[i]:
                    color = "red"

            plt.plot(depths, test_accs[i], label=exp_types[i], linewidth = 2, color = color)

    plt.legend()
    plt.xticks(depths)
    plt.title('Test Accuracies')
    plt.xlabel("Network Depth")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.title(title)

def plot_loss(train_loss, val_loss):
    


    plt.plot(range(1,len(train_loss)+1),
                train_loss,
                    label="train_loss")

    plt.plot(range(1,len(val_loss)+1),
                val_loss,
                    label="val_loss")
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
