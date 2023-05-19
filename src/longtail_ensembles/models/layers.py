"""
Create custom layers that can be used for interp ensembles. 
In order to construct resnet models that can be used for interpolation, we need four different kinds of basic layers: 

1. Conv2d     
2. BatchNorm2d
3. MaxPool2d
4. AdaptiveAvgPool2d
5. ReLU
6. Linear (to map flattened output)

Of these different kinds of layers, we need to change the ones which perform operations which intermix activities across different channels- namely Conv2d and Linear (used as a fully connected final layer). We also want to decouple parameters that scale linearly in the number of channels that they operate on- such as the bias of conv layers, batch norm, or the linear layer at the top of the network.   

"""
import math
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from scipy.signal import square
import torch
import numpy as np

use_cuda = torch.cuda.is_available()


def initialize_xavier(x):
    """xavier initialization for transformer attention layers. 

    """
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


def create_subnet_params_output_only(weights, nb_subnets):
    """Calculate subnet parameters, but make a mask that only cares about output channels. 
    """
    ## First check parity:
    weightshape = weights.shape
    assert weightshape[
        0] % nb_subnets == 0, "Out channel dim must be divisible by nb_subnets"
    ## Now compute block dimensions:
    blocks_perside = int(np.sqrt(nb_subnets))  ## assume perfect square
    blocklength = int(weightshape[0] / blocks_perside)
    blockheight = weightshape[1]

    ## Next create masks:
    masked_weights = {}
    for n in range(nb_subnets):
        ## Declare as boolean tensors:
        on_block = np.ones(
            (blocklength, blockheight, weightshape[2], weightshape[3]))
        blocks = [
            np.zeros(
                (blocklength, blockheight, weightshape[2], weightshape[3]))
            for ni in range(blocks_perside - 1)
        ]
        blocks.insert(n, on_block)
        reshaped_blocks = np.concatenate(
            [blocks[j] for j in range(blocks_perside)],
            axis=0)  # should generate rows of ordered blocks
        full_mask = torch.tensor(reshaped_blocks)
        masked_weights[n] = {"convweights": weights, "mask": full_mask}

    return masked_weights


def create_subnet_params(weights, nb_subnets):
    """Calculate parameters for each subnet. Done by chunking the weight matrix along the input and output channel dimensions. 

    :param weights: a tensor convolutional weights of shape (C_out,C_in,kernel0,kernel1)
    :param nb_subnets: a natural number indicating how we should split the convolutional weights. 
    :returns: a dictionary with keys (convweights,mask) where the first field gives the original convnet weights as parameters, and the second gives a tensor binary mask.     
    """
    ## First check parity:
    weightshape = weights.shape
    assert weightshape[
        0] % nb_subnets == 0, "Out channel dim must be divisible by nb_subnets"
    assert weightshape[
        1] % nb_subnets == 0, "In channel dim must be divisible by nb_subnets"
    ## Now compute block dimensions:
    blocks_perside = int(np.sqrt(nb_subnets))  ## assume perfect square
    blocklength = int(weightshape[0] / blocks_perside)
    blockheight = int(weightshape[1] / blocks_perside)

    ## Next create masks:
    masked_weights = {}
    for n in range(nb_subnets):
        ## Declare as boolean tensors:
        on_block = np.ones(
            (blocklength, blockheight, weightshape[2], weightshape[3]))
        blocks = [
            np.zeros(
                (blocklength, blockheight, weightshape[2], weightshape[3]))
            for ni in range(nb_subnets - 1)
        ]
        blocks.insert(n, on_block)
        reshaped_blocks = np.concatenate(
            [
                np.concatenate([
                    blocks[j + i * blocks_perside]
                    for j in range(blocks_perside)
                ],
                               axis=1) for i in range(blocks_perside)
            ],
            axis=0)  # should generate rows of ordered blocks
        full_mask = torch.tensor(reshaped_blocks)
        masked_weights[n] = {"convweights": weights, "mask": full_mask}

    return masked_weights


class Interp_Conv2d_factory(nn.Module):
    """This module initializes a standard `nn.Conv2d` layer, and also defines additional layers that act as "subnets", networks that preserve weights from different partitions of the input channels to different partitions of the output channels. This differs from group convolution in that group convolution only considers disjoint subsets of channels, while we consider all possible mappings between input and output channels.

    This class takes all the same arguments as nn.Conv2d, except for an additional argument `nb_subnets`. This argument defines the subnets we should use for downstream compute. Subnets are indexed by the block of convolutional weights that we select out.   
    NB: when we select a subnet index, this function initializes them with a separate set of biases than the ones that we use for the full network.  

    Usage: 
    ```
    ## one layer
    x = data
    lambda = 0.5
    conv_factory = Interp_Conv2d_factory(nb_subnets,*params)
    full_output = conv_factory.base_convnet(x)
    subnet_output = [conv_factory.subnets[i](x) for i in range(len(nb_subnets))]
    loss = lambda*rmse(x,full_output)+(1-lambda)*1/4(sum([rmse(x,subnet_output[i]) for i in range(4)]))
    loss.backwards()

    ```

    """

    def __init__(
        self,
        nb_subnets,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):

        super().__init__()
        ## record how many subnets you want:
        self.nb_subnets = nb_subnets
        ## initialize the full convnet
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype

        self.base_convnet = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias,
            self.padding_mode,
            self.device,
            self.dtype,
        )
        ## get the weights of the base convnet
        self.base_weights = self.base_convnet.weight
        self.base_bias = self.base_convnet.bias
        ## crete subnet weights:
        subnet_params = self.create_subnet_params(
            self.base_weights, nb_subnets
        )  ## does multiplication by zero, declaration of new biases.

        self.subnets = nn.ModuleList(self.create_subnets(subnet_params))

    def create_subnet_params(self, weights, nb_subnets):
        """Calculate parameters for each subnet. Done by chunking the weight matrix along the input and output channel dimensions. 

        :param weights: a tensor convolutional weights of shape (C_out,C_in,kernel0,kernel1)
        :param nb_subnets: a natural number indicating how we should split the convolutional weights. 
        :returns: a dictionary with keys (convweights,mask) where the first field gives the original convnet weights as parameters, and the second gives a tensor binary mask.     
        """
        ## First check parity:
        weightshape = weights.shape
        assert weightshape[
            0] % nb_subnets == 0, "Out channel dim must be divisible by nb_subnets"
        assert weightshape[
            1] % nb_subnets == 0, "In channel dim must be divisible by nb_subnets"
        ## Now compute block dimensions:
        blocks_perside = int(np.sqrt(nb_subnets))  ## assume perfect square
        blocklength = int(weightshape[0] / blocks_perside)
        blockheight = int(weightshape[1] / blocks_perside)

        ## Next create masks:
        masked_weights = {}
        for n in range(nb_subnets):
            ## Declare as boolean tensors:
            on_block = np.ones(
                (blocklength, blockheight, weightshape[2], weightshape[3]))
            blocks = [
                np.zeros(
                    (blocklength, blockheight, weightshape[2], weightshape[3]))
                for ni in range(nb_subnets - 1)
            ]
            blocks.insert(n, on_block)
            reshaped_blocks = np.concatenate(
                [
                    np.concatenate([
                        blocks[j + i * blocks_perside]
                        for j in range(blocks_perside)
                    ],
                                   axis=1) for i in range(blocks_perside)
                ],
                axis=0)  # should generate rows of ordered blocks
            full_mask = torch.tensor(reshaped_blocks)
            masked_weights[n] = {"convweights": weights, "mask": full_mask}

        return masked_weights

    def create_subnets(self, subnet_params):
        """Assign parameters to each subnet. 
        :param subnet_params: dictionary of weights for each subnet. each entry is a dictionary with fields {"convweights",torch.nn.Parameter,"mask",torch.tensor}, where the convweights are parameters of some other layer already. 
        """
        subnets = [
            Conv2d_subnet_layer(
                subnet_params[i],
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.bias,
                self.padding_mode,
                self.device,
                self.dtype,
            ) for i in range(self.nb_subnets)
        ]
        return subnets


class Conv2d_subnet_layer(nn.Conv2d):
    """Conv2d layer that can be initialized with custom weights and a permanent mask, generating inputs from the "subnet" consisting of just connections from one block of weights to another. 
    **NB** calling self.weight on this object will not show you the weights that are used for forward computation. The inputs are convolved against the weights defined by `self.get_mask`, which calculates `torch.mul(self.weight,self.mask)`.

    :param conv_weights: a dictionary with two entries: 1. "convweights" is a set of convolutional weights consistent with other standard `nn.Conv2d` which we will use to initialize these weights. 
    **All other params are identical to nn.Conv2d**. 

    """

    def __init__(self, conv_weights, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias, padding_mode, device,
                 dtype):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device,
                         dtype)
        if bias is not False:
            self.bias = bias
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        ## create masks:
        self.weight = conv_weights["convweights"]
        self.register_buffer("mask", conv_weights["mask"].type_as(self.weight))

    def get_masked(self):
        return torch.mul(self.weight, self.mask)

    def forward(self, input):
        return F.conv2d(input,
                        self.get_masked().float(),
                        bias=self.bias,
                        stride=self.stride,
                        padding=self.padding,
                        groups=self.groups,
                        dilation=self.dilation)


class AttnComparison(nn.Module):
    """Takes in sets of inputs (analogous to positions/tokens) as an arbitrary number of queries/keys, and determines a matrix of attention weightings of size nxn out of it using two linear projections, Q and K to turn these into queries and keys. The layer outputs a matrix of size n by n giving the attention of each query (rows) against each key (columns). Based on implementation here: https://github.com/tunz/transformer-pytorch/blob/e7266679f0b32fd99135ea617213f986ceede056/model/transformer.py#L60-L97 of multi-head attention. 

    :param in_features: The dimensionality we expect in incoming query/key inputs. 
    :param out_features: The dimensionality to which we project the queries and keys. 
    """

    def __init__(self, in_features, out_features):
        """
        """
        super().__init__()
        self.linear_q = nn.Linear(in_features, out_features, bias=False)
        self.linear_k = nn.Linear(in_features, out_features, bias=False)

        self.scale = out_features**(-0.5
                                    )  ## scaling for dot product attention.
        initialize_xavier(self.linear_q)
        initialize_xavier(self.linear_k)

    def forward(self, q, k):
        """Calculates scaled dot product attention between queries and keys after applying a learned linear transformation to each. 
        Assumes q is shape (batch,nb_queries,in_features), and k has shape (batch,nb_keys,in_features). 
        """
        q_proj = self.linear_q(q)  # batch,nb_queries,out_features
        k_proj = self.linear_k(k)  # batch,nb_keys,out_features

        ## transpose k
        k_transp = k_proj.transpose(1, 2)

        ## scale q before applying dot product.
        q_proj.mul_(self.scale)

        ## take dot product:
        scales = torch.matmul(q_proj, k_transp)  # batch,nb_queries,nb_keys
        weights = torch.softmax(scales, dim=2)

        return weights


class AttnMLPComparison(nn.Module):
    """Like the baseline AttnComparison function above, but with a single layer MLP instead of a linear projection into the attention space. 

    """

    def __init__(self, in_features, out_features, hidden_dim=1000):
        """
        """
        super().__init__()
        self.mlp_input_q = nn.Linear(in_features, hidden_dim)
        self.mlp_input_k = nn.Linear(in_features, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.mlp_output_q = nn.Linear(hidden_dim, out_features)
        self.mlp_output_k = nn.Linear(hidden_dim, out_features)

        self.scale = out_features**(-0.5
                                    )  ## scaling for dot product attention.
        initialize_xavier(self.mlp_input_q)
        initialize_xavier(self.mlp_input_k)
        initialize_xavier(self.mlp_output_q)
        initialize_xavier(self.mlp_output_k)

    def forward(self, q, k):
        """Calculates scaled dot product attention between queries and keys after applying a learned linear transformation to each. 
        Assumes q is shape (batch,nb_queries,in_features), and k has shape (batch,nb_keys,in_features). 
        """
        q_proj = self.mlp_output_q(self.relu(
            self.mlp_input_q(q)))  # batch,nb_queries,out_features
        k_proj = self.mlp_output_k(self.relu(
            self.mlp_input_k(k)))  # batch,nb_keys,out_features

        ## transpose k
        k_transp = k_proj.transpose(1, 2)

        ## scale q before applying dot product.
        q_proj.mul_(self.scale)

        ## take dot product:
        scales = torch.matmul(q_proj, k_transp)  # batch,nb_queries,nb_keys
        weights = torch.softmax(scales, dim=2)

        return weights


class PosEncodingsSin(nn.Module):
    """Positional encodings. Vary as sin waves in each model's outputs. Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html 

    Returns the input plus positional encodings 
    """

    def __init__(self, dmodel, dropout=0.1, max_len=None):
        super().__init__()
        if max_len is None:
            max_len = dmodel
        self.dropout = nn.Dropout(p=dropout)
        all_squares = [
            torch.tensor(np.sin(np.linspace(0, 4 * (1 + i) * np.pi,
                                            dmodel))).float()
            for i in range(max_len)
        ]
        pe = torch.stack(all_squares, axis=0).unsqueeze(1)

        #position = torch.arange(max_len).unsqueeze(1)
        #div_term = torch.exp(torch.arange(0, dmodel, 2) * (-math.log(10000.0) / dmodel))
        #pe = torch.zeros(max_len, 1, dmodel)
        #pe[:, 0, 0::2] = torch.sin(position * div_term)
        #pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x_transp = x.transpose(0, 1)

        x_pe = x_transp + self.pe[:x_transp.size(0)]
        x_pe_transp = x_pe.transpose(0, 1)

        return self.dropout(x_pe_transp)


class PosEncodingsSq(nn.Module):
    """Positional encodings. Vary as square waves in each model's outputs. Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html 

    Returns the input plus positional encodings 
    """

    def __init__(self, dmodel, dropout=0.1, max_len=None):
        super().__init__()
        if max_len is None:
            max_len = dmodel
        self.dropout = nn.Dropout(p=dropout)
        all_squares = [
            torch.tensor(-1 + 2 *
                         square(np.linspace(0, 2 *
                                            (1 + i) * np.pi, dmodel))).float()
            for i in range(max_len)
        ]
        pe = torch.stack(all_squares, axis=0).unsqueeze(1)

        #position = torch.arange(max_len).unsqueeze(1)
        #div_term = torch.exp(torch.arange(0, dmodel, 2) * (-math.log(10000.0) / dmodel))
        #pe = torch.zeros(max_len, 1, dmodel)
        #pe[:, 0, 0::2] = torch.sin(position * div_term)
        #pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x_transp = x.transpose(0, 1)

        x_pe = x_transp + self.pe[:x_transp.size(0)]
        x_pe_transp = x_pe.transpose(0, 1)

        return self.dropout(x_pe_transp)


class PosEncodings(nn.Module):
    """Positional encodings. Vary as sinusoids in position for each model dimension. Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html 

    Returns the input plus positional encodings 
    """

    def __init__(self, dmodel, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dmodel, 2) * (-math.log(10000.0) / dmodel))
        pe = torch.zeros(max_len, 1, dmodel)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x_transp = x.transpose(0, 1)

        x_pe = x_transp + self.pe[:x_transp.size(0)]
        x_pe_transp = x_pe.transpose(0, 1)

        return self.dropout(x_pe_transp)


class ChannelSwitcher(nn.Module):
    """Switches the first half and second half of channels given an activation of shape (batch, channels, height, width)

    :param channelnb: number of channels total
    """

    def __init__(self, channelnb):
        super().__init__()
        self.width = channelnb

    def forward(self, x):
        first_half = x[:, :int(self.width / 2), :, :]
        second_half = x[:, int(self.width / 2):, :, :]
        return torch.cat([second_half, first_half], axis=1)


class Interp_BatchNorm2d(nn.BatchNorm2d):
    """This special batchnorm layer swaps in different batch norm parameters if we use a subnet.   

    """


class Interp_Linear(nn.Linear):
    """This special linear layer users different weights when we use a subnet, because we separate mappings to the output layer.  

    """


class LogSoftmaxGroupLinear(nn.Linear):
    """This layer partitions the input features into separate groups that each project to all output channels. Before aggregating them as would be done in a normal matrix multiplication, this channel applies a logsoftmax transformation, and then takes an expectation.    
    """

    def __init__(self, in_features, out_features, groups, **kwargs):
        """This layer takes in all the normal arguments that nn.Linear takes, in addition to a "groups" argument that will split the input into `groups`. 

        :param in_features: number of input features. Must be divisible by group. 
        :param out_features: number of output features. 
        :param groups: number of groups to divide into. Must divide in_features. 
        :param bias = True:
        :param device = None:
        :param dtype = None:
        """
        super().__init__(in_features, out_features, **kwargs)
        self.groups = groups
        assert in_features % self.groups == 0
        "In features should be divisible by groups."
        self.lsm = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """   

        """
        x_chunked = torch.split(x, self.groups,
                                -1)  ## shape [(batch,groupsize)]
        weight_chunked = torch.split(self.weight, self.groups,
                                     1)  ## shape [(output,input)]

        actives = []
        for x_g, weight_g in zip(x_chunked, weight_chunked):
            actives.append(
                self.lsm(F.linear(x_g, weight_g))
            )  ## actives is a list of activations, each with size (batch,*,output)
        mean_active = sum(actives) / len(actives)
        return mean_active
