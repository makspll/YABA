from typing import Callable, Optional
from torch.utils.data import Dataset
from common.yaml_addons import YAMLObjectUninitializedFiltered
from torchvision.datasets import CIFAR10,MNIST,CIFAR100


class Subset(Dataset):
    def __init__(self,dataset,indices):
        self.dataset = dataset
        self.indices = indices
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class YAMLDataset(YAMLObjectUninitializedFiltered):
    def create(self, train, root, transform, target_transform):
        return super().create(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform)


class CIFAR10(YAMLDataset):
    yaml_tag='!DCIFAR10'
    yaml_fields=['download']
    yaml_class_target=CIFAR10

class CIFAR100(YAMLDataset):
    yaml_tag='!DCIFAR100'
    yaml_fields=['download']
    yaml_class_target=CIFAR100

class MNIST(YAMLDataset):
    yaml_tag='!DMNIST'
    yaml_fields=['download']
    yaml_class_target=MNIST
