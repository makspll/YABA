import torch 
import torch.nn.functional as F
import torch.nn as nn
from model.blocks import BNResBottleneckBlock, BNResidualBlock, EntryBlock


class Resnet(nn.Module):
    def __init__(self, 
                 input_shape, 
                 num_output_classes, 
                 num_filters,
                 num_blocks_per_stage, 
                 num_stages, 
                 use_bias=False, 
                 processing_block_type=BNResidualBlock,
                 dimensionality_reduction_block_type=BNResBottleneckBlock):
        """
        Initializes a convolutional network module
        :param input_shape: The shape of the tensor to be passed into this network
        :param num_output_classes: Number of output classes
        :param num_filters: Number of filters per convolutional layer
        :param num_blocks_per_stage: Number of blocks per "stage". Each block is composed of 2 convolutional layers.
        :param num_stages: Number of stages in a network. A stage is defined as a sequence of layers within which the
        data dimensionality remains constant in the spacial axis (h, w) and can change in the channel axis. After each stage
        there exists a dimensionality reduction stage, composed of two convolutional layers and an avg pooling layer.
        :param use_bias: Whether to use biases in our convolutional layers
        :param processing_block_type: Type of processing block to use within our stages
        :param dimensionality_reduction_block_type: Type of dimensionality reduction block to use after each stage in our network
        """
        super().__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_blocks_per_stage = num_blocks_per_stage
        self.num_stages = num_stages
        self.processing_block_type = processing_block_type
        self.dimensionality_reduction_block_type = dimensionality_reduction_block_type

        # build the network
        self.build_module()

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        self.layer_dict = nn.ModuleDict()
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        x = torch.zeros((self.input_shape))  # create dummy inputs to be used to infer shapes of layers

        out = x
        self.layer_dict['input_conv'] = EntryBlock(input_shape=out.shape, num_filters=self.num_filters,
                                                                kernel_size=3, padding=1, bias=self.use_bias,
                                                                dilation=1)
        out = self.layer_dict['input_conv'].forward(out)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        for i in range(self.num_stages):  # for number of layers times
            for j in range(self.num_blocks_per_stage):
                self.layer_dict['block_{}_{}'.format(i, j)] = self.processing_block_type(input_shape=out.shape,
                                                                                         num_filters=self.num_filters,
                                                                                         bias=self.use_bias,
                                                                                         kernel_size=3, dilation=1,
                                                                                         padding=1)
                out = self.layer_dict['block_{}_{}'.format(i, j)].forward(out)
            self.layer_dict['reduction_block_{}'.format(i)] = self.dimensionality_reduction_block_type(
                input_shape=out.shape,
                num_filters=self.num_filters, bias=True,
                kernel_size=3, dilation=1,
                padding=1,
                reduction_factor=2)
            out = self.layer_dict['reduction_block_{}'.format(i)].forward(out)

        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view(out.shape[0], -1)
        self.logit_linear_layer = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.num_output_classes,
                                            bias=True)
        out = self.logit_linear_layer(out)  # apply linear layer on flattened inputs
        return out

    def forward(self, x):
        """
        Forward propages the network given an input batch
        :param x: Inputs x (b, c, h, w)
        :return: preds (b, num_classes)
        """
        out = x
        out = self.layer_dict['input_conv'].forward(out)
        for i in range(self.num_stages):  # for number of layers times
            for j in range(self.num_blocks_per_stage):
                out = self.layer_dict['block_{}_{}'.format(i, j)].forward(out)
            out = self.layer_dict['reduction_block_{}'.format(i)].forward(out)

        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view(out.shape[0], -1)  # flatten outputs from (b, c, h, w) to (b, c*h*w)
        out = self.logit_linear_layer(out)  # pass through a linear layer to get logits/preds

        return out

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except Exception as E:
                print(E)
                pass

        self.logit_linear_layer.reset_parameters()




