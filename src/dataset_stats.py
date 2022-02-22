import sys 
from args import CALC_PARSER
from common.yaml_addons import get_loader
from dataset import YAMLDataset
from exceptions import ConfigurationException
from runner import Config
import yaml 
import numpy as np 

if __name__ == "__main__":
    args = CALC_PARSER.parse_args()

    ## load config
    config = None
    with open(args.config, 'r') as f:
        config = yaml.load(f,get_loader())
    
    with open(args.out,'w') as out:
        if args.mode == "per pixel mean":
            ds : YAMLDataset = config["dataset"]
            ds = ds.create(config.get('train',True),args.datasets,None, None)

            summed = np.zeros_like(np.moveaxis(np.asarray(ds[0][0]),-1,0))
            for x,y in ds:
                summed += np.moveaxis(np.asarray(x),-1,0)
            summed = summed / np.array(len(ds),dtype=float)

            out.writelines([f"{str(x)}\n" for x in summed.flatten(order='C')]) # row major order, default but just gotta make sure ya kno
        if args.mode == "stratified split":

            from sklearn.model_selection import train_test_split

            ds : YAMLDataset = config["dataset"]
            ds = ds.create(config.get('train',True),args.datasets,None, None)


            X = []
            y = []

            for idx,(_,label) in enumerate(ds):
                X.append(idx)
                y.append(label)

            X_train,X_test,y_train,y_test = train_test_split(X,y,
                train_size=args.split_size,
                stratify=y)

            out.writelines([f"{str(x)}\n" for x in X_train])