
from common import err_if_none_arg
from vis import plot_bn_vs_other_gradient_magnitudes, plot_accuracy
from args import GRAPH_PARSER
from os.path import join,isdir,isfile,basename
import os 
import pandas as pd
import logging 
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    args = GRAPH_PARSER.parse_args()

    logging.basicConfig(level=args.loglevel)

    
    err_if_none_arg(args.experiment_name,"experiment_name")
    err_if_none_arg(args.graph_type,"graph_type")
    err_if_none_arg(args.out,"out")

    experiment_root = join(args.experiments,args.experiment_name)

    weights_root = join(experiment_root,"weights")
    logs_root = join(experiment_root,"logs")
    
    if args.graph_type == "gradient_magnitude":
        # get all 
        epoch_checkpoints = sorted([join(weights_root,x) for x in os.listdir(weights_root) if isfile(join(weights_root,x)) and x.startswith("epoch")]
            ,key=lambda x: int(''.join([c for c in basename(x) if c.isdigit()])))

        plot_bn_vs_other_gradient_magnitudes(epoch_checkpoints)

    if args.graph_type == "accuracy":
        # get all 
        csv_values  = (pd.read_csv(join(experiment_root), "epoch_stats.csv"))
        train_acc   = csv_values["train_acc"].tolist()
        val_acc     = csv_values["val_acc"].tolist()
        

        plot_accuracy(train_acc, val_acc)

    if args.graph_type == "loss":
        # get all 
        csv_values  = (pd.read_csv(join(experiment_root), "epoch_stats.csv"))
        train_loss   = csv_values["train_loss"].tolist()
        val_loss     = csv_values["val_loss"].tolist()
        

        plot_accuracy(train_loss, val_loss)
    
    plt.savefig(args.out)
    if args.show:
        plt.show()
        
 