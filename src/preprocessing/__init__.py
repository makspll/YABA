from common.yaml_addons import YAMLObjectFiltered
from torchvision import transforms

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


class RandomAffine(YAMLObjectFiltered, transforms.RandomAffine):
    yaml_tag="!PPRandomAffine"
    yaml_fields=["degrees","translate", "scale", "shear", "interpolation", "fill", "fillcolor", "resample"]