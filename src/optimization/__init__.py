from common.yaml_addons import YAMLObjectUninitializedFiltered
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

class YAMLOptimizer(YAMLObjectUninitializedFiltered):
    def create(self,params):
        return super().create(params=params)

class YAMLScheduler(YAMLObjectUninitializedFiltered):
    def create(self, optimizer):
        return super().create(optimizer=optimizer)

class AdamOptimizer(YAMLOptimizer):
    yaml_tag='!OAdam'
    yaml_fields=["lr","betas","eps","weight_decay","amsgrad"]
    yaml_class_target=Adam

class SGDOptimizer(YAMLOptimizer):
    yaml_tag='!OSGD'
    yaml_fields=["lr","momentum","dampening","weight_decay","nesterov"]
    yaml_class_target=SGD

class CosineAnnealingLR(YAMLScheduler):
    yaml_tag='!SCosineAnnealing'
    yaml_fields=["T_max","eta_min","last_epoch","verbose"]
    yaml_class_target=CosineAnnealingLR

class MultiStepLR(YAMLScheduler):
    yaml_tag='!SMultiStep'
    yaml_fields=["milestones","gamma","last_epoch","verbose"]
    yaml_class_target=MultiStepLR