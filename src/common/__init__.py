from enum import Enum
import torch 
import numpy as np
import random 

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
