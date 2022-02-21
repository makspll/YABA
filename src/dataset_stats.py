import sys 
from args import CALC_PARSER
from common.yaml_addons import get_loader
from dataset import YAMLDataset
from exceptions import ConfigurationException
from runner import Config
import yaml 
import numpy as np 

# CALC_PARSER = argparse.ArgumentParser()

# CALC_PARSER.add_argument("--config",
#     help="the yaml config file to parse looking for dataset object or others")

# CALC_PARSER.add_argument("--out",
#     help="the path to output file",
#     default="calc_out.txt")

# CALC_PARSER.add_argument("--mode",
#     choices=["per_pixel_mean"])
if __name__ == "__main__":
    args = CALC_PARSER.parse_args()

    ## load config
    config = None
    with open(args.config, 'r') as f:
        config = yaml.load(f,get_loader())
    
    out = open(args.out,'w')
    if args.mode == "per pixel mean":
        ds : YAMLDataset = config["dataset"]
        ds = ds.create(config.get('train',True),args.datasets,None, None)

        summed = np.zeros_like(np.moveaxis(np.asarray(ds[0][0]),-1,0))
        for x,y in ds:
            summed += np.moveaxis(np.asarray(x),-1,0)
        print(summed.shape)
        summed = summed / np.array(len(ds),dtype=float)

        out.writelines([f"{str(x)}\n" for x in summed.flatten(order='C')]) # row major order, default but just gotta make sure ya kno