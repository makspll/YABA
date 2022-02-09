from dataclasses import dataclass
from typing import Callable, Optional
from torch.utils.data import Dataset
from common import StringableEnum
from torchvision.datasets import CIFAR10,MNIST


class Subset(Dataset):
    def __init__(self,dataset,indices):
        self.dataset = dataset
        self.indices = indices
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class DatasetClass(StringableEnum):
    CIFAR_10 = CIFAR10
    MNIST = MNIST