# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

# Code was based on https://github.com/ChengpengChen/RepGhost/blob/main/model/repghost.py
# reference: https://arxiv.org/abs/2211.06088

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingNormal


__all__ = [
    'repghostnet_0_5x',
    'repghostnet_0_58x',
    'repghostnet_0_8x',
    'repghostnet_1_0x',
    'repghostnet_1_11x',
    'repghostnet_1_3x',
    'repghostnet_1_5x',
    'repghostnet_2_0x',
]


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace=False):
    if inplace:
        return paddle.clip(x + 3.0, min=0, max=6) / 6.0
    else:
        return F.relu6(x + 3.0) / 6.0


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


class SqueezeExcite(nn.Layer):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=nn.ReLU,
        gate_fn=hard_sigmoid,
        divisor=4,
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible(
            (reduced_base_chs or in_chs) * se_ratio, divisor,
        )
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv_reduce = nn.Conv2D(in_chs, reduced_chs, 1, bias_attr=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2D(reduced_chs, in_chs, 1, bias_attr=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class RepConvBNLayer(nn.Layer):
    """ 
    MobileOne building block.
    This block has a multi-branched architecture at train-time
    and plain-CNN style architecture at inference time
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 num_conv_branches=1):
        """ 
        Construct a MobileOneBlock module.
        Args:
            in_channels(int): Number of channels in the input.
            out_channels(int): Number of channels produced by the block.
            kernel_size(int): Size of the convolution kernel.
            stride(int): Stride size.
            groups(int): Group number.
            use_se(bool): Whether to use SE-ReLU activations.
            num_conv_branches(int): Number of linear conv branches.
        """
        super(RepConvBNLayer, self).__init__()
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

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
        

    def forward(self, x):
        # Inference mode forward pass.
        if hasattr(self, "reparam_conv"):
            return self.reparam_conv(x)

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Other branches
        out = identity_out
        for idx in range(self.num_conv_branches):
            out += self.rbr_conv[idx](x)

        return out

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
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

    def _get_kernel_bias(self):
        """ 
        Method to obtain re-parameterized kernel and bias.
        """
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

        kernel_final = kernel_conv + kernel_identity
        bias_final = bias_conv + bias_identity
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


class RepGhostModule(nn.Layer):
    def __init__(
        self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True):
        super(RepGhostModule, self).__init__()
        init_channels = oup
        new_channels = oup

        self.primary_conv = nn.Sequential(
            ConvBNLayer(inp, init_channels, kernel_size, stride),
            nn.ReLU() if relu else nn.Sequential(),
        )

        self.fusion_bn = nn.BatchNorm2D(init_channels)

        self.cheap_operation = RepConvBNLayer(
                init_channels,
                new_channels,
                dw_size,
                1,
                groups=init_channels)

        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.Sequential()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return self.relu(x2)


class RepGhostBottleneck(nn.Layer):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        se_ratio=0.0,
        shortcut=True,
        deploy=False,
    ):
        super(RepGhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.enable_shortcut = shortcut
        self.in_chs = in_chs
        self.out_chs = out_chs

        # Point-wise expansion
        self.ghost1 = RepGhostModule(
            in_chs,
            mid_chs,
            relu=True,
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = ConvBNLayer(mid_chs, mid_chs, dw_kernel_size, stride, groups=mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = RepGhostModule(
            mid_chs,
            out_chs,
            relu=False,
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                ConvBNLayer(in_chs, in_chs, dw_kernel_size, stride, groups=in_chs),
                ConvBNLayer(in_chs, out_chs, 1, 1)
            )

    def forward(self, x):
        residual = x

        # 1st repghost bottleneck
        x1 = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x1)
        else:
            x = x1

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd repghost bottleneck
        x = self.ghost2(x)
        if not self.enable_shortcut and self.in_chs == self.out_chs and self.stride == 1:
            return x
        return x + self.shortcut(residual)


class RepGhostNet(nn.Layer):
    def __init__(
        self,
        cfgs,
        class_num=1000,
        width=1.0,
        dropout=0.2,
        shortcut=True,
        deploy=False,
    ):
        super(RepGhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout
        self.class_num = class_num

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = ConvBNLayer(3, output_channel, 3, 2)
        self.act1 = nn.ReLU()
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = RepGhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(
                    block(
                        input_channel,
                        hidden_channel,
                        output_channel,
                        k,
                        s,
                        se_ratio=se_ratio,
                        shortcut=shortcut,
                        deploy=deploy
                    ),
                )
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width * 2, 4)
        stages.append(
            nn.Sequential(
                ConvBNLayer(input_channel, output_channel, 1, 1),
                nn.ReLU()
            ),
        )
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        self.global_pool = nn.AdaptiveAvgPool2D((1, 1))
        self.conv_head = nn.Conv2D(
            input_channel, output_channel, 1, 1, 0, bias_attr=True,
        )
        self.act2 = nn.ReLU()
        self.classifier = nn.Linear(output_channel, class_num)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

    def convert_to_deploy(self):
        repghost_model_convert(self, do_copy=False)


def repghost_model_convert(model, save_path=None, do_copy=True):
    """
    taken from from https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        paddle.save(model.state_dict(), save_path)
    return model


def repghostnet(enable_se=True, **kwargs):
    """
    Constructs a RepGhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 8, 16, 0, 1]],
        # stage2
        [[3, 24, 24, 0, 2]],
        [[3, 36, 24, 0, 1]],
        # stage3
        [[5, 36, 40, 0.25 if enable_se else 0, 2]],
        [[5, 60, 40, 0.25 if enable_se else 0, 1]],
        # stage4
        [[3, 120, 80, 0, 2]],
        [
            [3, 100, 80, 0, 1],
            [3, 120, 80, 0, 1],
            [3, 120, 80, 0, 1],
            [3, 240, 112, 0.25 if enable_se else 0, 1],
            [3, 336, 112, 0.25 if enable_se else 0, 1],
        ],
        # stage5
        [[5, 336, 160, 0.25 if enable_se else 0, 2]],
        [
            [5, 480, 160, 0, 1],
            [5, 480, 160, 0.25 if enable_se else 0, 1],
            [5, 480, 160, 0, 1],
            [5, 480, 160, 0.25 if enable_se else 0, 1],
        ],
    ]

    return RepGhostNet(cfgs, **kwargs)


def repghostnet_0_5x(**kwargs):
    return repghostnet(width=0.5, **kwargs)


def repghostnet_wo_0_5x(**kwargs):
    return repghostnet(width=0.5, shortcut=False, **kwargs)


def repghostnet_0_58x(**kwargs):
    return repghostnet(width=0.58, **kwargs)


def repghostnet_0_8x(**kwargs):
    return repghostnet(width=0.8, **kwargs)


def repghostnet_1_0x(**kwargs):
    return repghostnet(width=1.0, **kwargs)


def repghostnet_1_11x(**kwargs):
    return repghostnet(width=1.11, **kwargs)


def repghostnet_1_3x(**kwargs):
    return repghostnet(width=1.3, **kwargs)


def repghostnet_1_5x(**kwargs):
    return repghostnet(width=1.5, **kwargs)


def repghostnet_2_0x(**kwargs):
    return repghostnet(width=2.0, **kwargs)