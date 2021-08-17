#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import collections
import itertools
import six
import math
import sys
import warnings
from functools import partial, reduce

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle import framework
from paddle.device import get_device, get_cudnn_version
from paddle.nn import functional as F
from paddle.nn import initializer as I
from paddle.nn import Layer, LayerList
from paddle.fluid.layers import utils
from paddle.fluid.layers.utils import map_structure, flatten, pack_sequence_as
from paddle.fluid.data_feeder import convert_dtype
from paddle import _C_ops
__all__ = []

class RseNetUnit(Layer):
    r"""
    ******Temporary version******.
    ResNetUnit is designed for optimize the performence by using cudnnv8 API.
    """

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                #  padding=0,
                #  dilation=1,
                #  groups=1,
                 ele_count=1,
                 conv_format='NCHW',
                 bn_format='NCHW',
                 act=None,
                 fused_add=False,
                 has_shortcut=False,
                 name=None):
        super(RseNetUnit, self).__init__()
        self._in_channels = num_channels
        self._out_channels = num_filters
        self._kernel_size = filter_size
        self._stride = stride
        self._padding = (filter_size - 1) // 2
        self._dilation = 1
        self._groups = 1
        self._conv_format = conv_format
        self._bn_format = bn_format
        self._act = act

        # check format
        valid_format = {'NHWC', 'NCHW'}
        if conv_format not in valid_format:
            raise ValueError(
                "conv_format must be one of {}, but got conv_format='{}'".
                format(valid_format, conv_format))
        if bn_format not in valid_format:
            raise ValueError(
                "bn_format must be one of {}, but got bn_format='{}'".
                format(valid_format, bn_format))

    def forward(self, inputs):
        if fluid.framework.in_dygraph_mode():
            out = _C_ops.resnet_unit()
        else:
            inputs = {
            }
            attrs = {
            }

            outputs = {
            }

            self._helper.append_op(
                type="resnet_unit", inputs=inputs, outputs=outputs, attrs=attrs)

        return out