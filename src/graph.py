
from common import err_if_none_arg
from vis import plot_bn_vs_other_gradient_magnitudes, plot_accuracy, plot_loss
from args import GRAPH_PARSER
from os.path import join,isdir,isfile,basename
import os 
import pandas as pd
import logging 
import matplotlib.pyplot as plt 

def plot_gradient_magnitude(weights_root, out):
    # get all 
    epoch_checkpoints = sorted([join(weights_root,x) for x in os.listdir(weights_root) if isfile(join(weights_root,x)) and x.startswith("epoch")]
        ,key=lambda x: int(''.join([c for c in basename(x) if c.isdigit()])))

    plot_bn_vs_other_gradient_magnitudes(epoch_checkpoints)
    out = join(out, "gradient_magnitudes.png")
    plt.savefig(out)

def plot_acc_curve(stats, out):
    # get all 
    csv_values  = pd.read_csv(stats)
    train_acc   = csv_values["train_acc"].tolist()
    val_acc     = csv_values["val_acc"].tolist()
    

    plot_accuracy(train_acc, val_acc)
    out = join(out, "accuracy.png")
    plt.savefig(out)

def plot_loss_curve(stats, out):
    # get all 
    csv_values  =  pd.read_csv(stats)
    train_loss   = csv_values["train_loss"].tolist()
    val_loss     = csv_values["val_loss"].tolist()
    

    plot_loss(train_loss, val_loss)
    out = join(out, "loss.png")
    plt.savefig(out)
    
if __name__ == "__main__":
    args = GRAPH_PARSER.parse_args()

    logging.basicConfig(level=args.loglevel)

    
    err_if_none_arg(args.experiment_name,"experiment_name")
    err_if_none_arg(args.graph_type,"graph_type")
    #err_if_none_arg(args.out,"out")

    experiment_root = join(args.experiments,args.experiment_name)

    weights_root = join(experiment_root,"weights")
    logs_root = join(experiment_root,"logs")
    stats = join(logs_root, "epoch_stats.csv")
    graphs_root = join("graphs", args.experiment_name)
    if(not os.path.exists(graphs_root)):
    	os.mkdir(graphs_root)
 
    
    if args.graph_type == "gradient_magnitude":
        plot_gradient_magnitude(weights_root, graphs_root)

    if args.graph_type == "accuracy":
        plot_acc_curve(stats, graphs_root)

    if args.graph_type == "loss":
        plot_loss_curve(stats, graphs_root)
        

    if args.graph_type == "all":
        plot_gradient_magnitude(weights_root, graphs_root)
        plot_acc_curve(stats, graphs_root)
        plot_loss_curve(stats, graphs_root)
        
    
    if args.show:
        plt.show()




