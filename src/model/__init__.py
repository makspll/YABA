from functools import partial
from common import StringableEnum
from .baseline import Resnet

class ModelClass(StringableEnum):
    RESNET_200 = Resnet 
