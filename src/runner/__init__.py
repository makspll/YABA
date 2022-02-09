
from copy import deepcopy
from csv import DictWriter
from dataclasses import dataclass
from datetime import date
from logging import Logger
import logging
import sys
from typing import Callable, Dict, List

import numpy as np
import common
from exceptions import ConfigurationException
from model import ModelClass
from dataset import DatasetClass,Subset
import torch 
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml 
import os 
from os.path import join,isdir
from torchvision import transforms

class Config():
    """ object representing the config file for all experiments, converts certain
        string format args to their respective enums,
        let's us have static type hints, woohoo """

    def __init__(self, config : Dict) -> None:
        self.experiment_name : str = config['experiment_name']

        try:
            self.dataset : Callable[[], Dataset] = DatasetClass.from_str(config["dataset"]).value
        except Exception as E:
            raise ConfigurationException("dataset",f"{E.__class__.__name__}:{E}")
        
        try:
            self.model : Callable[[],nn.Module] = ModelClass.from_str(config["model"]).value
        except Exception as E:
            raise ConfigurationException("model",f"{E.__class__.__name__}:{E.with_traceback(None)}")

        self.gpus : List[int] = config['gpus']
        self.batch_size : int = config['batch_size']
        self.learning_rate : float = config['learning_rate']
        self.validation_list : List[int] = config['validation_list']
        self.epochs : int = config['epochs']

    def to_yaml(self,f):
        copy = deepcopy(self)
        copy.dataset = DatasetClass.to_str_from_value(self.dataset)
        copy.model = ModelClass.to_str_from_value(self.model)

        yaml.dump(vars(copy),f)

class ExperimentRunner():
    def __init__(self,
        config : Config,
        root : str,
        resume : bool,
        datasets : str) -> None:

        self.resume = resume
        self.root = root 
        self.config = config 
        self.epoch = 1
        self.best_val_epoch = -1
        self.best_val_acc = -1

        ## sort out gpus and model   
        self.model = config.model()

        if not torch.cuda.is_available():
            raise ConfigurationException("gpus", "No gpus are available, but some were requested. Stop using Windows you twat")

        self.device = torch.cuda.current_device()
        if torch.cuda.device_count() >= len(self.config.gpus):
            self.model.to(self.device)
            if len(self.config.gpus) > 1: # wrap around in parallelizer
                self.model = nn.DataParallel(self.model)
        

        ## define preprocessing transforms

        # TODO: make this customizable via arguments
        self.transforms = transforms.Compose([
            transforms.ToTensor(), # TODO: change this default to something sensible
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
        ])

        self.target_transforms = transforms.Compose([
            
        ])

        ## sort out data
        common_kwargs = {
            "root": datasets,
            "transform":self.transforms,
            "target_transform":self.target_transforms
        }

        dataset_training = self.config.dataset(train=True,**common_kwargs)
        training_list = [x for x in range(len(dataset_training)) if x not in config.validation_list]
        dataset_training = Subset(dataset_training,training_list)

        dataset_validation = self.config.dataset(train=True,**common_kwargs)
        dataset_validation = Subset(dataset_validation,indices=config.validation_list)
                

        dataset_test = self.config.dataset(train=False,**common_kwargs),


        common_kwargs = {
            "batch_size":config.batch_size,
            "num_workers":4
        }

        self.training_data = DataLoader(
            dataset_training,**common_kwargs)
        
        self.validation_data = DataLoader(
            dataset_validation,**common_kwargs)

        self.testing_data = DataLoader(
            dataset_test,**common_kwargs)
        
        ## sort out gradient descent parameters via arguments
        self.optimizer = Adam(self.model.parameters(),
            lr=config.learning_rate) # TODO: optimizer selection

        self.loss_function  = nn.CrossEntropyLoss().to(self.device) # TODO: loss function selection via arguments

        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=0.00002) # TODO: lr scheduler selection via arguments

        ## create directories or retrieve existing ones
        if not isdir(self.root):
            os.mkdir(self.root)

        self.experiment_root = join(self.root,self.config.experiment_name)

        self.log_dir = join(self.experiment_root,"logs")
        self.config_dir = join(self.experiment_root,"config")
        self.weights_dir = join(self.experiment_root,"weights")

        if isdir(self.experiment_root):
            if self.resume:
                self.load(join(self.weights_dir, self.checkpoint_name(last=True)))
            else:
                raise Exception("Experiment already exists, but 'resume' flag is not set.")

        os.makedirs(self.log_dir,exist_ok=True)
        os.makedirs(self.config_dir,exist_ok=True)
        os.makedirs(self.weights_dir,exist_ok=True)


        ## write down config (might be changed since resume, so name epoch too)
        with open(join(self.config_dir,f"config_{self.epoch}.yaml"),'w') as f:
            self.config.to_yaml(f)

        ## setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.FileHandler(join(self.log_dir,f"{date.today()}.log")))
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

    def start(self):
        for i in range(self.epoch,self.config.epochs):
            self.epoch = i 
            
            epoch_stats = {
                "train_acc" : [],
                "train_loss" : [],
                "val_acc" : [],
                "val_loss" : [],
            }

            for x,y in self.training_data:
                loss, acc = self.iter(x,y,self.epoch,False)
                epoch_stats["train_loss"].append(loss)
                epoch_stats["train_acc"].append(acc)

            for x,y in self.validation_data:
                loss, acc = self.iter(x,y,self.epoch, True)
                epoch_stats["val_acc"].append(acc)
                epoch_stats["val_loss"].append(acc)

            # convert stats to averages
            epoch_stats = {k:np.mean(v) for k,v in epoch_stats.items()}

            self.logger.info(f"({self.epoch}/{self.config.epochs}) - {', '.join([f'{k}:{v}' for k,v in epoch_stats.items()])}")

            if epoch_stats['val_acc'] > self.best_val_acc:
                self.best_val_acc = epoch_stats['val_acc']
                self.best_val_epoch = self.epoch

            # TODO: track epoch time and log val acc + train acc after each epoch

            self.dump_stats(self.epoch, epoch_stats)
            self.checkpoint(self.checkpoint_name(self.epoch))
            self.checkpoint(self.checkpoint_name(last=True)) # overwrite existing last 

        ## load best model and run on test set

        self.load(join(self.weights_dir, self.checkpoint_name(epoch_no=self.best_val_epoch)))
        
        test_stats = {
            "test_acc" : [],
            "test_loss" : []
        }
        for x,y in self.testing_data:
            loss, acc = self.iter(x,y,self.best_val_epoch,True)
            test_stats["test_acc"].append(acc)
            test_stats["test_loss"].append(loss)

        test_stats = {k:np.mean(v) for k,v in test_stats.items()}
        self.dump_stats(self.best_val_epoch,final_test_stats=test_stats)
    
    def dump_stats(self,epoch: int, epoch_stats : Dict= None, final_test_stats : Dict = None):
        # TODO: more stats options, feature flags to enable tracking of certain things
        if epoch_stats:
            with open(join(self.log_dir,"epoch_stats.csv"),'w') as f:
                w = DictWriter(f,epoch_stats.keys())
                if epoch == 1:
                    w.writeheader()

                w.writerow(epoch_stats)

        if final_test_stats:
            with open(join(self.log_dir,"test_stats.csv"),'w') as f:
                w = DictWriter(f,epoch_stats.keys())
                w.writeheader()

                w.writerow(epoch_stats)
 

    def checkpoint_name(self, epoch_no: int = -1, last=False) -> str:
        if last:
            return f"epoch_last.pt"
        else:
            return f"epoch_{epoch_no}.pt"

    def iter(self, x : torch.Tensor, y : torch.Tensor, epoch, eval_mode):
        
        if eval_mode:
            self.model.eval()
        else:
            self.model.train()

        x,y = x.to(device=self.device),y.to(device=self.device)
        out = self.model(x)

        loss = self.loss_function(out,y)

        if not eval_mode:
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()
        
        accuracy = torch.sum(y == out)

        loss = loss.cpu().detach().numpy()

        return loss,accuracy
    

    def checkpoint(self,name:str):
        
        torch.save({ # TODO: store gradients as well
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_epoch': self.best_val_epoch, 
        }, join(self.weights_dir,name))

    def load(self, path :str):
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_epoch = checkpoint['best_val_epoch']
