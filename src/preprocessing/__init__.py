import torch
from common.yaml_addons import YAMLObjectFiltered
from torchvision import transforms
from functools import reduce
from operator import mul

class CenterCrop(YAMLObjectFiltered,transforms.CenterCrop):
    yaml_tag='!PPCenterCrop'
    yaml_fields=["size"]

class RandomCrop(YAMLObjectFiltered,transforms.RandomCrop):
    yaml_tag='!PPRandomCrop'
    yaml_fields=["size","padding","pad_if_needed","fill","padding_mode"]


class RandomHorizontalFlip(YAMLObjectFiltered, transforms.RandomHorizontalFlip):
    yaml_tag="!PPRandomHorizontalFlip"
    yaml_fields=["p"]

class ToTensor(YAMLObjectFiltered, transforms.ToTensor):
    yaml_tag="!PPToTensor"
    yaml_fields=[]


class Normalize(YAMLObjectFiltered, transforms.Normalize):
    yaml_tag="!PPNormalize"
    yaml_fields=["mean","std","inplace"]

class ElementwiseZeroing(YAMLObjectFiltered):
    yaml_tag="!PPElementwiseZeroing"
    yaml_fields=["mean_file","input_shape"]

    def __init__(self, mean_file,input_shape) -> None:
        super(torch.nn.Module).__init__()

        self.mean_file = mean_file 
        self.input_shape = input_shape

        with open(self.mean_file) as f:
            self.mean = torch.Tensor([float(x) for x in f.readlines()])
            expected_len = reduce(mul,self.input_shape)
            if len(self.mean) != expected_len:
                raise Exception(f"Provided mean file '{self.mean_file}' contains {len(self.mean)} elements, but with input shape {self.input_shape} it should contain {expected_len} elements.")
            
            self.mean = self.mean.reshape(self.input_shape) 

    def __call__(self, img: torch.Tensor) ->  torch.Tensor:
        o = img - self.mean.to(img.device) # should broadcast over channels 
        assert(o.shape == img.shape)
        return o
        

class RandomAffine(YAMLObjectFiltered, transforms.RandomAffine):
    yaml_tag="!PPRandomAffine"
    yaml_fields=["degrees","translate", "scale", "shear", "interpolation", "fill", "fillcolor", "resample"]