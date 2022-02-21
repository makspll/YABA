
from csv import DictWriter
from datetime import date
import logging
from typing import Dict, List, Union
import numpy as np
from exceptions import ConfigurationException
from model import YAMLModel
from dataset import Subset, YAMLDataset
import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml 
import os 
from os.path import join,isdir,isfile,realpath,dirname
from torchvision import transforms
from optimization import YAMLOptimizer, YAMLScheduler
from .trackers import Tracker
from common import get_random_state_dicts, set_random_state_dict, set_seed
import re
import time

class Config():
    """ object representing the config file for all experiments, converts certain
        string format args to their respective enums,
        let's us have static type hints, woohoo """

    def __init__(self, config : Dict) -> None:
        self.experiment_name : str = config['experiment_name']
        self.gpus : List[int] = config['gpus']
        self.fast_mode : bool = config.get('fast_mode',False)
        self.batch_size : int = config['batch_size']
        self.validation_list : Union[List[int],str] = config['validation_list']
        self.epochs : int = config['epochs']
        self.trackers : List[Tracker] = config['trackers']
        self.seed : int = config['seed']
        self.transforms : List = config.get('transforms',[])
        self.transforms_test : List = config.get('transforms_test',[])
        self.target_transforms : List = config.get('target_transforms',[])
        self.target_transforms_test : List = config.get('target_transform_test',[])
        self.freeze_parameter_list : List[str] = config.get('freeze_parameter_list',[])
        self.model : YAMLModel = config['model']
        self.optimizer: YAMLOptimizer = config['optimizer']
        self.scheduler: YAMLScheduler = config['scheduler']
        self.dataset : YAMLDataset = config['dataset']
        self.init_weights = config.get('init_weights', False)
        
    def to_yaml(self,f):
        yaml.dump(vars(self),f)

    def __repr__(self) -> str:
        return "\n".join([f"\t{k}:{self.__getattribute__(k)}" for k in vars(self)])

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

        self.logger = logging.getLogger(__name__)

        ## sort out gpus and model   
        self.model = self.config.model.create()
        if(config.init_weights):
            self.model.apply(self.init_weights)

        
        # freeze parameters
        frozen = []
        compiled_regex = [re.compile(x) for x in self.config.freeze_parameter_list]
        for r in compiled_regex:
            for n,v in self.model.named_parameters():
                if r.search(n):
                    v.requires_grad = False
                    frozen.append(n)

        self.logger.info(f"Froze parametrs: {frozen}")

        # log model stats
        self.log_model_stats()

        # attach trackers
        for t in self.config.trackers:
            t.attach(self.model)


        if len(self.config.gpus) > 0:
            if not torch.cuda.is_available():
                raise ConfigurationException("gpus", "No gpus are available, but some were requested. Stop using Windows you twat")

            self.device = torch.cuda.current_device()
            if torch.cuda.device_count() >= len(self.config.gpus):
                if len(self.config.gpus) > 1: # wrap around in parallelizer
                    self.model = nn.DataParallel(self.model)

            for i in range(torch.cuda.device_count()):
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name(i)}")

        else:
            self.device = 'cpu'

        self.model.to(self.device)


        ## sort out data
        common_kwargs_test = {
            "root": datasets,
            "transform": transforms.Compose(self.config.transforms_test),
            "target_transform" : transforms.Compose(self.config.target_transforms_test),
        }
        common_kwargs_train = {
            "root": datasets,
            "transform": transforms.Compose(self.config.transforms),
            "target_transform" : transforms.Compose(self.config.target_transforms),
        }

        # read validation list, either already a list or path to one
        list_path = join(dirname(realpath(__file__)),'..','..','lists')
        val_list = self.config.validation_list 
        if isinstance(self.config.validation_list,str):
            val_list = [int(x) for x in open(join(list_path,self.config.validation_list)).readlines()]

        

        dataset_training = self.config.dataset.create(True,*common_kwargs_train.values())
        training_list = [x for x in range(len(dataset_training)) if x not in val_list]
        dataset_training = Subset(dataset_training,training_list)

        dataset_validation = self.config.dataset.create(True,*common_kwargs_test.values())
        dataset_validation = Subset(dataset_validation,indices=val_list)
                
        dataset_test = self.config.dataset.create(False,*common_kwargs_test.values())
        self.logger.info(f"Using {len(training_list)} training samples, and {len(dataset_validation)} validation samples.")
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

        ## create directories or retrieve existing ones
        if not isdir(self.root):
            os.mkdir(self.root)

        self.experiment_root = join(self.root,self.config.experiment_name)

        self.log_dir = join(self.experiment_root,"logs")
        self.config_dir = join(self.experiment_root,"config")
        self.weights_dir = join(self.experiment_root,"weights")


        
        ## sort out gradient descent parameters via arguments
        self.optimizer = self.config.optimizer.create(self.model.parameters())
        self.loss_function  = nn.CrossEntropyLoss().to(self.device)
        self.lr_scheduler = self.config.scheduler.create(self.optimizer)

        ## load checkpoint

        if isdir(self.experiment_root):
            if self.resume:
                self.load(join(self.weights_dir, self.checkpoint_name(last=True)))
                self.epoch += 1 # we store last processed epoch, start at next one
            else:
                raise Exception("Experiment already exists, but 'resume' flag is not set.")
        else:
            set_seed(self.config.seed,self.config.fast_mode) 

        os.makedirs(self.log_dir,exist_ok=True)
        os.makedirs(self.config_dir,exist_ok=True)
        os.makedirs(self.weights_dir,exist_ok=True)



        ## write down config (might be changed since resume, so name epoch too)        
        with open(join(self.config_dir,f"config_{self.epoch}.yaml"),'w') as f:
            self.config.to_yaml(f)

        ## setup logger
        file_handler = logging.FileHandler(join(self.log_dir,f"{date.today()}.log"))
        detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)


    def log_model_stats(self):
        params = list(self.model.named_parameters())
        non_trainable_params = sum([x.numel() for (k,x) in params if not x.requires_grad])
        trainable_params = sum([x.numel() for (k,x) in params if x.requires_grad ])
        conv_layers = len([1 for (k,x) in params if "conv" in k and not "bias" in k])

        self.logger.info(f"Layer names containing 'conv' : {conv_layers}")
        self.logger.info(f"Non-trainable parameters (Thousands): {non_trainable_params/1e3}")
        self.logger.info(f"Trainable parameters (Thousands): {trainable_params/1e3}")
        self.logger.info(f"Total Parameters (Thousands): {(trainable_params + non_trainable_params)/1e3}")


    def start(self):

        self.logger.info(f"Begun training at epoch {self.epoch}.")


        
        for i in range(self.epoch,self.config.epochs+1):
            self.epoch = i 
            epoch_time = time.time()

            epoch_stats = {
                "train_acc" : [],
                "train_loss" : [],
                "val_acc" : [],
                "val_loss" : [],
            }


            epoch_training_time = time.time()
            training_load_time = time.time()
            for x,y in self.training_data:
                training_load_time = time.time() - training_load_time


                loss, acc = self.iter(x,y,self.epoch,False)
                epoch_stats["train_loss"].append(loss)
                epoch_stats["train_acc"].append(acc)

            epoch_training_time = time.time() - epoch_training_time

            post_train_trackers = {}
            for t in self.config.trackers:
                post_train_trackers[t.key] = t.post_train_iter_hook()

            epoch_validation_time = time.time()

            for x,y in self.validation_data:
                loss, acc = self.iter(x,y,self.epoch, True)
                epoch_stats["val_acc"].append(acc)
                epoch_stats["val_loss"].append(loss)

            epoch_validation_time = time.time() - epoch_validation_time

            # convert stats to averages
            epoch_stats = {k:np.mean(v) for k,v in epoch_stats.items()}



            if epoch_stats['val_acc'] > self.best_val_acc:
                self.best_val_acc = epoch_stats['val_acc']
                self.best_val_epoch = self.epoch

            # TODO: track epoch time and log val acc + train acc after each epoch

            dump_time = time.time()

            self.dump_stats("epoch_stats",write_header=self.epoch==1,**epoch_stats)
            self.checkpoint(self.checkpoint_name(self.epoch),**post_train_trackers)
            self.checkpoint(self.checkpoint_name(last=True),**post_train_trackers) 

            dump_time = time.time() - dump_time

            epoch_time = time.time() - epoch_time


            self.logger.info(f"Epoch {self.epoch}/{self.config.epochs} : {', '.join([f'{k}:{v:.3f}' for k,v in epoch_stats.items()])} ({epoch_time:.3f}s, ETA:{epoch_time * (self.config.epochs+1 - self.epoch) / 60 / 60:.3f}h)")
            self.logger.debug(f"Time split: train:{epoch_training_time:.2f}s, load_train:{training_load_time:.2f}  val:{epoch_validation_time:.2f}s, stats:{dump_time:.2f}s)")
            
            self.lr_scheduler.step()
            self.logger.debug(f"Next learning rate: {self.optimizer.param_groups[0]['lr']:.4f}")

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
        self.dump_stats("final_test_stats",write_header=True,**test_stats)

        self.logger.info("Completed training.")


    def dump_stats(self,name,write_header=False, **kwargs):
        # TODO: more stats options, feature flags to enable tracking of certain things
        with open(join(self.log_dir,f"{name}.csv"),'a') as f:
            w = DictWriter(f,kwargs.keys())
            if write_header == 1:
                w.writeheader()
            strings = {k:f"{v:{len(k)}.3f}" for k,v in kwargs.items()}

            w.writerow(strings)

 

    def checkpoint_name(self, epoch_no: int = -1, last=False) -> str:
        """ generates the epoch filename for the given epoch number, 
        if last is true, will find the last epoch available in the weights directory """

        if last:
            last_epoch_file_name = sorted(
                    [x for x in os.listdir(self.weights_dir) if isfile(join(self.weights_dir,x))],
                    key=lambda x: int(''.join([c for c in x if c.isdigit()])))
            if last_epoch_file_name:
                last_epoch_file_name = last_epoch_file_name[-1]
            else:
                raise Exception("Requested last epoch name, but weight directory is empty, did you try to resume empty experiment ?")

            assert(last_epoch_file_name.endswith('.pt'))

            return last_epoch_file_name
        else:
            return f"epoch_{epoch_no}.pt"

    def iter(self, x : torch.Tensor, y : torch.Tensor, epoch, eval_mode):
        
        if eval_mode:
            self.model.eval()
        else:
            self.optimizer.zero_grad()
            self.model.train()

        x,y = x.to(device=self.device),y.to(device=self.device)
        out = self.model(x)


        loss = self.loss_function(out,y)

        if not eval_mode:
            loss.backward()

            self.optimizer.step()

        predicted = torch.argmax(out.data, 1)  # get argmax of predictions (output is logit)
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        loss = loss.cpu().detach().numpy()

        return loss,accuracy
    

    def checkpoint(self,name:str, **kwargs):
        model = self.model
        if isinstance(self.model,nn.DataParallel):
            model = self.model.module 

        torch.save({ # TODO: store gradients as well
            'epoch': self.epoch,
            'model_state': model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_epoch': self.best_val_epoch,
            'all_seed_states': get_random_state_dicts(),
            **kwargs 
        }, join(self.weights_dir,name))

    def load(self, path :str):
        model = self.model
        if isinstance(self.model,nn.DataParallel):
            model = self.model.module 

        checkpoint = torch.load(path)
        
        model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_epoch = checkpoint['best_val_epoch']
        set_random_state_dict(checkpoint['all_seed_states'])

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

