from enum import Enum
import sys
from typing import Dict, Union
import torch 
import numpy as np
import random 
import logging 

logger = logging.getLogger(__name__)

def err_if_none_arg(arg,key):
    """if arg is none quits program and shows error message

    Args:
        arg ([type]): the value of key or none
        key ([type]): the key name 
    """
    if not arg:
        logger.error(f"{key}: argument missing, needs to be specified in config or argument")
        sys.exit(1)

def override_if_not_none(arg : Union[None,object], key, config : Dict):
    """Overrides key in the config if arg is not None, if no key in config and arg is none
       quits the program with error message    

    Args:
        arg ([type]): the value or none
        key ([type]): the key name 
        config ([type]): config in dictionary form
    """
    if arg:
        config[key] = arg
    else:
        logger.error(config.get(key,None),key)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 
    # for cuda
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

def get_random_state_dicts():

    return {
        "torch": torch.get_rng_state(),
        "np": np.random.get_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state_all(),
    }

def set_random_state_dict(dict):
    torch.set_rng_state(dict['torch'])
    np.random.set_state(dict['np'])
    random.setstate(dict['python'])
    torch.cuda.set_rng_state_all(dict['cuda'])


class StringableEnum(Enum):
    """ class which enables easy string to Enum, Enum value to string conversion """
    @classmethod
    def from_str(cls, label: str) -> "StringableEnum":
        return cls[label.strip().upper().replace(' ','_')]

    @classmethod
    def to_str_from_value(cls, value) -> str:
        return cls(value).name.lower().replace('_',' ')
