from functools import partial
from common import StringableEnum
from .Resnet import Resnet
from .VGG import VGG

class ModelClass(StringableEnum):
    RESNET = Resnet 
    VGG = VGG
