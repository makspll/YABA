import torch 
import torch.nn.functional as F
import torch.nn as nn



class EntryBlock(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation):
        super(EntryBlock, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.bias = bias
        self.dilation = dilation

        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_0'].forward(out)
        self.layer_dict['bn_0'] = nn.BatchNorm2d(num_features=out.shape[1])
        out = F.leaky_relu(self.layer_dict['bn_0'].forward(out))


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

        
    def forward(self, x):
        out = x

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(self.layer_dict['bn_0'].forward(out))

        return out


    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        for item in self.layer_dict.children():
            item.reset_parameters()

        self.logits_linear_layer.reset_parameters()


class BNResidualBlock(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation):
        super().__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.bias = bias
        self.dilation = dilation

        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=False,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_0'].forward(out)

        # note bias is not used in conv_0, since batch norm discards it anyway
        self.layer_dict['bn_0'] = nn.BatchNorm2d(num_features=out.shape[1])
        out = self.layer_dict['bn_0'](out)

        out = F.leaky_relu(out)


        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=False,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_1'].forward(out)
        self.layer_dict['bn_1'] = nn.BatchNorm2d(num_features=out.shape[1])
        out = self.layer_dict['bn_1'](out)

        out = x + out


        out = F.leaky_relu(out)

    def forward(self, x):
        out = x

        out = self.layer_dict['conv_0'].forward(out)
        out = self.layer_dict['bn_0'](out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_1'].forward(out)
        out = self.layer_dict['bn_1'](out)        
        out = x + out  # skip connection
        out = F.leaky_relu(out)

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



class BNResBottleneckBlock(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation, reduction_factor):
        super().__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.bias = bias
        self.dilation = dilation
        self.reduction_factor = reduction_factor
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=False,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_0'].forward(out)

        self.layer_dict['bn_0'] = nn.BatchNorm2d(num_features=out.shape[1])
        out = self.layer_dict['bn_0'](out)
        
        out = F.leaky_relu(out)


        out = F.avg_pool2d(out, self.reduction_factor)

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=False,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_1'].forward(out)

        # we require a 1x1 connection with stride 2 to downsample the residual appropriately (no bias)
        self.layer_dict['conv_1_skip'] = nn.Conv2d(in_channels=x.shape[1], out_channels=self.num_filters, bias=False,
                                              kernel_size=1, dilation=self.dilation,
                                              padding=0, stride=2)
        self.layer_dict['bn_1'] = nn.BatchNorm2d(num_features=out.shape[1])
        out = self.layer_dict['bn_1'](out)

        out = self.layer_dict['conv_1_skip'](x) + out 


        
        out = F.leaky_relu(out)




    def forward(self, x):
        out = x

        out = self.layer_dict['conv_0'].forward(out)
        out = self.layer_dict['bn_0'](out)
        out = F.leaky_relu(out)

        out = F.avg_pool2d(out, self.reduction_factor)

        out = self.layer_dict['conv_1'].forward(out)
        out = self.layer_dict['bn_1'](out)
        out = self.layer_dict['conv_1_skip'](x) + out 
        out = F.leaky_relu(out)

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

class VGG(nn.Module):
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




