
from common import err_if_none_arg
from vis import plot_bn_vs_other_gradient_magnitudes, plot_accuracy, plot_bn_vs_other_gradient_magnitudes_relative, plot_bn_vs_other_parameter_magnitudes, plot_loss, plot_final_accuracy
from args import GRAPH_PARSER
from os.path import join,isdir,isfile,basename
import os 
import pandas as pd
import logging 
import matplotlib.pyplot as plt 
import re

def plot_gradient_magnitude(weights_root, out):
    # get all 
    epoch_checkpoints = sorted([join(weights_root,x) for x in os.listdir(weights_root) if isfile(join(weights_root,x)) and x.startswith("epoch")]
        ,key=lambda x: int(''.join([c for c in basename(x) if c.isdigit()])))

    plot_bn_vs_other_gradient_magnitudes(epoch_checkpoints)
    out = join(out, "gradient_magnitudes.pdf")
    plt.savefig(out)
    plt.clf()

def plot_gradient_magnitude_rel(weights_root, out):
    # get all 
    epoch_checkpoints = sorted([join(weights_root,x) for x in os.listdir(weights_root) if isfile(join(weights_root,x)) and x.startswith("epoch")]
        ,key=lambda x: int(''.join([c for c in basename(x) if c.isdigit()])))

    plot_bn_vs_other_gradient_magnitudes_relative(epoch_checkpoints)
    out = join(out, "gradient_magnitudes_relative.pdf")
    plt.savefig(out)
    plt.clf()

def plot_parameter_magnitude(weights_root, out):
    # get all 
    epoch_checkpoints = sorted([join(weights_root,x) for x in os.listdir(weights_root) if isfile(join(weights_root,x)) and x.startswith("epoch")]
        ,key=lambda x: int(''.join([c for c in basename(x) if c.isdigit()])))

    plot_bn_vs_other_parameter_magnitudes(epoch_checkpoints)
    out = join(out, "parameter_magnitudes.pdf")
    plt.savefig(out)
    plt.clf()

def plot_acc_curve(stat, out):
    # get all 
    csv_values  = pd.read_csv(stat)
    train_acc   = csv_values["train_acc"].tolist()
    val_acc     = csv_values["val_acc"].tolist()
        

    plot_accuracy(train_acc, val_acc)
    out = join(out, "accuracy.pdf")
    plt.savefig(out)
    plt.clf()

def plot_final_accs_curve(stats, out, exp_types, title):
    all_accs = []
    for stat in stats:
        exp_accs = []
        
        for s in stat:
            try:
                csv_values  = pd.read_csv(s)
                exp_accs.append(csv_values["test_acc"].tolist()[0])
            except:
                exp_accs.append(0)
        all_accs.append(exp_accs)
    
    plot_final_accuracy(all_accs, exp_types, title)
    plt.savefig(out)
    plt.clf()


def plot_loss_curve(stat, out):
    # get all 
    csv_values  =  pd.read_csv(stat)
    train_loss   = csv_values["train_loss"].tolist()
    val_loss     = csv_values["val_loss"].tolist()
    

    plot_loss(train_loss, val_loss)
    out = join(out, "loss.pdf")
    plt.savefig(out)
    plt.clf()


if __name__ == "__main__":
    args = GRAPH_PARSER.parse_args()

    logging.basicConfig(level=args.loglevel)

    
    #err_if_none_arg(args.experiment_name,"experiment_name")
    err_if_none_arg(args.graph_type,"graph_type")

    weights_roots = []
    logs_roots = []
    final_stats_base = []
    final_stats_frankle = []
    weights_roots = []
    graphs_roots = []
    exp_types_base = []
    exp_types_frankle = []
    rootdir = args.experiments
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            
            exp_type = (os.path.basename(os.path.normpath(d)))
            exp_type = re.findall('([a-zA-Z\_ ]*)\d*.*',exp_type)
            exp_type = (exp_type[0][:-1])
            
            if "frankle" in exp_type:
                exp_types_frankle.append(exp_type)
                
            else:
                exp_types_base.append(exp_type)

    exp_folders = []

    exp_types_base = list(dict.fromkeys(exp_types_base))
    exp_types_frankle = list(dict.fromkeys(exp_types_frankle))

    for exp_type in exp_types_base:
        exp_56 = exp_type + "_56_cifar10"
        exp_110 = exp_type + "_110_cifar10"
        exp_218 = exp_type + "_218_cifar10"
        exps = [exp_56, exp_110, exp_218]
        stats = []
            
        stats.append("experiments/"+ exp_56 + "/logs/final_test_stats.csv")
        stats.append("experiments/"+ exp_110 + "/logs/final_test_stats.csv")
        stats.append("experiments/"+ exp_218 + "/logs/final_test_stats.csv")
        final_stats_base.append(stats)

    for exp_type in exp_types_frankle:
        exp_56 = exp_type + "_56_cifar10"
        exp_110 = exp_type + "_110_cifar10"
        exp_218 = exp_type + "_218_cifar10"
        exps = [exp_56, exp_110, exp_218]
        stats = []
            
        stats.append("experiments/"+ exp_56 + "/logs/final_test_stats.csv")
        stats.append("experiments/"+ exp_110 + "/logs/final_test_stats.csv")
        stats.append("experiments/"+ exp_218 + "/logs/final_test_stats.csv")
        final_stats_frankle.append(stats)
            

    experiment_root = join(args.experiments,args.experiment_name)
    
    weights_root = join(experiment_root,"weights")
    logs_root = join(experiment_root,"logs")
    stats = join(logs_root, "epoch_stats.csv")
    graphs_root = join("graphs", args.experiment_name)

    
    if(not os.path.exists(graphs_root)):
        os.mkdir(graphs_root)
 
    if args.graph_type == "gradient_magnitude" or args.graph_type == "all":
        plot_gradient_magnitude(weights_root, graphs_root)

    if args.graph_type == "gradient_magnitude_rel" or args.graph_type == "all":
        plot_gradient_magnitude_rel(weights_root, graphs_root)
    
    if args.graph_type == "parameter_magnitude" or args.graph_type == "all":
        plot_parameter_magnitude(weights_root, graphs_root)

    if args.graph_type == "accuracy"  or args.graph_type == "all":
        plot_acc_curve(stats, graphs_root)

    if args.graph_type == "loss"  or args.graph_type == "all":
        plot_loss_curve(stats, graphs_root)

    if args.experiment_name == "all" and args.graph_type == "all_acc":
        plot_final_accs_curve(final_stats_base, join(graphs_root, "resnet_accs.pdf"), exp_types_base, "Accuracies for ResNet Models")
        plot_final_accs_curve(final_stats_frankle, join(graphs_root, "frankle_accs.pdf"), exp_types_frankle, "Accuracies for Frankle (Frozen) Models")

    if args.show:
        plt.show()




