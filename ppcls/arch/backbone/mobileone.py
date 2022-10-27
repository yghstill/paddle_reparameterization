# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This Code was based on https://github.com/apple/ml-mobileone/blob/main/mobileone.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingNormal


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 filter_size,
                 stride,
                 groups=1):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)
        self.bn = nn.BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SEBlock(nn.Layer):
    """ 
    Squeeze and Excite module.
    """
    def __init__(self, in_channels, rd_ratio=0.0625):
        """ 
        Construct a Squeeze and Excite Module.
        Args:
            in_channels(int): Number of input channels.
            rd_ratio(float): Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias_attr=False)
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias_attr=False)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.act = nn.ReLU()
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, inputs):
        """ Apply forward pass. """
        outputs = self.avg_pool(inputs)
        outputs = self.reduce(outputs)
        outputs = self.act(outputs)
        outputs = self.expand(outputs)
        outputs = self.hardsigmoid(outputs)
        return paddle.multiply(x=inputs, y=outputs)


class MobileOneBlock(nn.Layer):
    """ 
    MobileOne building block.
    This block has a multi-branched architecture at train-time
    and plain-CNN style architecture at inference time
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 inference_mode=False,
                 use_se=False,
                 num_conv_branches=1):
        """ 
        Construct a MobileOneBlock module.
        Args:
            in_channels(int): Number of channels in the input.
            out_channels(int): Number of channels produced by the block.
            kernel_size(int): Size of the convolution kernel.
            stride(int): Stride size.
            groups(int): Group number.
            inference_mode(bool): If True, instantiates model in inference mode.
            use_se(bool): Whether to use SE-ReLU activations.
            num_conv_branches(int): Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()
        self.activation = nn.ReLU()

        # Re-parameterizable skip connection
        self.rbr_skip = nn.BatchNorm2D(
            num_features=in_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0))
        ) if in_channels == out_channels and self.stride == 1 else None

        # Re-parameterizable conv branches
        self.rbr_conv = nn.LayerList()
        for _ in range(self.num_conv_branches):
            self.rbr_conv.append(
                ConvBNLayer(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    stride=self.stride,
                    groups=self.groups))

        # Re-parameterizable scale branch
        self.rbr_scale = None
        if kernel_size > 1:
            self.rbr_scale = ConvBNLayer(
                self.in_channels,
                self.out_channels,
                1,
                stride=self.stride,
                groups=self.groups)

    def forward(self, x):
        # Inference mode forward pass.
        if hasattr(self, "reparam_conv"):
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for idx in range(self.num_conv_branches):
            out += self.rbr_conv[idx](x)

        return self.activation(self.se(out))

    def convert_to_deploy(self):
        """
        Re-parameterize multi-branched architecture used at training
        time to obtain a plain CNN-like structure for inference.
        """
        if hasattr(self, 'reparam_conv'):
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(self.kernel_size - 1) // 2,
            groups=self.groups)
        self.reparam_conv.weight.set_value(kernel)
        self.reparam_conv.bias.set_value(bias)

        # Delete un-used branches
        self.__delattr__('rbr_conv')
        if hasattr(self, 'rbr_scale'):
            self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

    def _get_kernel_bias(self):
        """ 
        Method to obtain re-parameterized kernel and bias.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size. 1x1->3x3
            padding_size = self.kernel_size // 2
            kernel_scale = paddle.nn.functional.pad(
                kernel_scale,
                [padding_size, padding_size, padding_size, padding_size])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(
                self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0

        if isinstance(branch, nn.LayerList):
            fused_kernels = []
            fused_bias = []
            for block in branch:
                kernel = block.conv.weight
                running_mean = block.bn._mean
                running_var = block.bn._variance
                gamma = block.bn.weight
                beta = block.bn.bias
                eps = block.bn._epsilon

                std = (running_var + eps).sqrt()
                t = (gamma / std).reshape((-1, 1, 1, 1))

                fused_kernels.append(kernel * t)
                fused_bias.append(beta - running_mean * gamma / std)

            return sum(fused_kernels), sum(fused_bias)

        elif isinstance(branch, ConvBNLayer):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            input_dim = self.in_channels if self.kernel_size == 1 else 1
            kernel_value = paddle.zeros(
                shape=[
                    self.in_channels, input_dim, self.kernel_size,
                    self.kernel_size
                ],
                dtype='float32')
            if self.kernel_size > 1:
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, (self.kernel_size - 1) // 2,
                                 (self.kernel_size - 1) // 2] = 1
            elif self.kernel_size == 1:
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 0, 0] = 1
            else:
                raise ValueError("Invalid kernel size recieved!")
            kernel = paddle.to_tensor(kernel_value, place=branch.weight.place)
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))

        return kernel * t, beta - running_mean * gamma / std


class MobileOneNet(nn.Layer):
    """ 
    PaddlePaddle implementation of `An Improved One millisecond Mobile Backbone`
    https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 class_num=1000,
                 num_blocks_per_stage=[2, 8, 10, 1],
                 width_multipliers=[0.75, 1.0, 1.0, 2.0],
                 use_se=False,
                 num_conv_branches=1):
        """
        Construct MobileOne model.
        Args:
            class_num: Number of classes in the dataset.
            num_blocks_per_stage: List of number of blocks per stage.
            width_multipliers: List of width multiplier for blocks in a stage.
            use_se: Whether to use SE-ReLU activations.
            num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneNet, self).__init__()
        assert len(width_multipliers) == 4
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        # Build stages
        self.stage0 = MobileOneBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2)
        self.stage1 = self._make_stage(
            int(64 * width_multipliers[0]),
            num_blocks_per_stage[0],
            num_se_blocks=0)
        self.stage2 = self._make_stage(
            int(128 * width_multipliers[1]),
            num_blocks_per_stage[1],
            num_se_blocks=0)
        self.stage3 = self._make_stage(
            int(256 * width_multipliers[2]),
            num_blocks_per_stage[2],
            num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0)
        self.stage4 = self._make_stage(
            int(512 * width_multipliers[3]),
            num_blocks_per_stage[3],
            num_se_blocks=num_blocks_per_stage[3] if use_se else 0)
        self.gap = nn.AdaptiveAvgPool2D(output_size=1)
        self.linear = nn.Linear(int(512 * width_multipliers[3]), class_num)

    def _make_stage(self, planes, num_blocks, num_se_blocks=0):
        """ 
        Build a stage of MobileOne model.
        Args:
            planes(int): Number of output channels.
            num_blocks(int): Number of blocks in this stage.
            num_se_blocks(int): Number of SE blocks in this stage.
        """
        # Get strides for all layers
        strides = [2] + [1] * (num_blocks - 1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot "
                                 "exceed number of layers.")
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # Depthwise conv
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=stride,
                    groups=self.in_planes,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches))
            # Pointwise conv
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=1,
                    groups=1,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = paddle.flatten(x, start_axis=1)
        x = self.linear(x)
        return x


PARAMS = {
    "s0": {
        "width_multipliers": (0.75, 1.0, 1.0, 2.0),
        "num_conv_branches": 4
    },
    "s1": {
        "width_multipliers": (1.5, 1.5, 2.0, 2.5)
    },
    "s2": {
        "width_multipliers": (1.5, 2.0, 2.5, 4.0)
    },
    "s3": {
        "width_multipliers": (2.0, 2.5, 3.0, 4.0)
    },
    "s4": {
        "width_multipliers": (3.0, 3.5, 3.5, 4.0),
        "use_se": True
    },
}


def MobileOne(class_num=1000, variant="s0"):
    """
    Get MobileOne model.
    Args:
        class_num(int): Number of classes in the dataset.
        variant(str): Which type of model to generate.
    """
    variant_params = PARAMS[variant]
    return MobileOneNet(class_num=class_num, **variant_params)


def MobileOne_S0(pretrained=False, use_ssld=False, **kwargs):
    variant_params = PARAMS["s0"]
    model = MobileOneNet(**variant_params, **kwargs)
    return model


def reparameterize_model(model):
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for layer in model.sublayers():
        if hasattr(layer, 'convert_to_deploy'):
            layer.convert_to_deploy()
    return model