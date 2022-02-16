import torch.nn as nn
import torch.nn.functional as F
import torch

class EntryBlock(nn.Module):
    def __init__(self,in_channels, num_filters, kernel_size, padding, bias, dilation):
        super(EntryBlock, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.dilation = dilation
        self.in_channels = in_channels

        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_filters, bias=self.bias,
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
    def __init__(self, in_channels, num_filters, kernel_size, padding, bias, dilation):
        super().__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.padding = padding
        self.bias = bias
        self.dilation = dilation

        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_filters, bias=False,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        # note bias is not used in conv_0, since batch norm discards it anyway
        self.layer_dict['bn_0'] = nn.BatchNorm2d(num_features=self.num_filters)


        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters, bias=False,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        self.layer_dict['bn_1'] = nn.BatchNorm2d(num_features=self.num_filters)


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
    def __init__(self, in_channels, num_filters, kernel_size, padding, bias, dilation, reduction_factor):
        super().__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.padding = padding
        self.bias = bias
        self.dilation = dilation
        self.reduction_factor = reduction_factor
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_filters, bias=False,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_0'].forward(out)

        self.layer_dict['bn_0'] = nn.BatchNorm2d(num_features=self.num_filters)
        out = self.layer_dict['bn_0'](out)
        
        out = F.leaky_relu(out)


        out = F.avg_pool2d(out, self.reduction_factor)

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters, bias=False,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_1'].forward(out)

        # we require a 1x1 connection with stride 2 to downsample the residual appropriately (no bias)
        self.layer_dict['conv_1_skip'] = nn.Conv2d(in_channels=x.shape[1], out_channels=self.num_filters, bias=False,
                                              kernel_size=1, dilation=self.dilation,
                                              padding=0, stride=2)
        self.layer_dict['bn_1'] = nn.BatchNorm2d(num_features=self.num_filters)
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
            