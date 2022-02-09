from functools import partial
from common import StringableEnum
from torchvision.models import resnet152 

class ModelClass(StringableEnum):
    RESNET_200 = partial(resnet152) # enums are funky about storing functions as values
