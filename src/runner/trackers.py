from typing import Union
import torch.nn as nn
from common.yaml_addons import YAMLObjectFiltered

class Tracker(YAMLObjectFiltered):
    """ implements hooks which return something to keep track of (or None if hook is not supported) at each calling point
    """
    yaml_tag = '!Tracker'
    yaml_fields = ['key']

    def __init__(self, key):
        self.model = None
        self.key = key

    def attach(self, model):
        self.model = model 

    def post_train_iter_hook(self) -> Union[object,None]:
        return None


class GradientsTracker(Tracker):
    yaml_tag = '!GradientsTracker'

    def post_train_iter_hook(self):
        gradients = {}
        for k,v in self.model.named_parameters():
            gradients[k] = v.grad
        return gradients
