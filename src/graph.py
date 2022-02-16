
from common import err_if_none_arg
from vis import plot_bn_vs_other_gradient_magnitudes, plot_gradient_flow
from args import GRAPH_PARSER
from os.path import join,isfile,basename
import os 
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
    epoch_checkpoints = sorted([join(weights_root,x) for x in os.listdir(weights_root) if isfile(join(weights_root,x)) and x.startswith("epoch")]
        ,key=lambda x: int(''.join([c for c in basename(x) if c.isdigit()])))
    if args.graph_type == "gradient_magnitude":
        plot_bn_vs_other_gradient_magnitudes(epoch_checkpoints,join(logs_root,"epoch_stats.csv"))
    if args.graph_type == 'gradient_flow':
        plot_gradient_flow(epoch_checkpoints)

    plt.savefig(args.out)
    if args.show:
        plt.show()
        
 