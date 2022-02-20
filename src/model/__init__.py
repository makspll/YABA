from functools import partial
from common import StringableEnum
from .Resnet import Resnet
from .VGG import VGG
from common.yaml_addons import YAMLObjectUninitializedFiltered

class YAMLModel(YAMLObjectUninitializedFiltered):
    def create(self):
        return super().create()


class Resnet(YAMLModel):
    yaml_tag='!MResnet'
    yaml_fields=["layers","block","num_output_classes","zero_init_residual","groups","width_per_group","replace_stride_with_dilation","sparse_bn","norm_layer"]
    yaml_class_target=Resnet 

class VGG(YAMLModel):
    yaml_tag='!MVGG'
    yaml_fields=["input_shape","num_output_classes","num_filters","num_blocks_per_stage","num_stages","use_bias","processing_block_type","dimensionality_reduction_block_type"]
    yaml_class_target=VGG 

