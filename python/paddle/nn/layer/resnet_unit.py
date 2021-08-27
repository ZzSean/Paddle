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
__all__ = ['res_unit']


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
                 ele_count=1,
                 momentum=0.9,
                 eps=1e-5,
                 conv_format='NHWC',
                 bn_format='NHWC',
                 act=None,
                 fused_add=False,
                 has_shortcut=False,
                 filter_x_attr=None,
                 filter_z_attr=None,
                 name=None):
        super(RseNetUnit, self).__init__()
        self._in_channels = num_channels
        self._out_channels = num_filters
        self._stride = stride
        self._dilation = 1
        self._kernel_size = utils.convert_to_list(filter_size, 2, 'kernel_size')
        self._padding = (filter_size - 1) // 2
        self._groups = 1
        self._ele_count = ele_count
        self._momentum = momentum
        self._eps = eps
        self._conv_format = conv_format
        self._bn_format = bn_format
        self._act = act
        self._fused_add = fused_add
        self._has_shortcut = has_shortcut
        self._filter_x_attr = filter_x_attr
        self._filter_z_attr = filter_z_attr

        # check format
        valid_format = {'NHWC'}
        if conv_format not in valid_format:
            raise ValueError(
                "conv_format must be one of {}, but got conv_format='{}'".
                format(valid_format, conv_format))
        if bn_format not in valid_format:
            raise ValueError(
                "bn_format must be one of {}, but got bn_format='{}'".format(
                    valid_format, bn_format))

        def _get_default_param_initializer():
            filter_elem_num = np.prod(self._kernel_size) * self._in_channels
            std = (2.0 / filter_elem_num)**0.5
            return Normal(0.0, std)

        # initial filter
        filter_shape = [num_filters, num_channels, filter_size, filter_size]
        self.filter_x = self.create_parameter(
            shape=filter_shape,
            attr=self._filter_x_attr,
            default_initializer=_get_default_param_initializer())
        if has_shortcut:
            self.filter_z = self.create_parameter(
                shape=filter_shape,
                attr=self._filter_z_attr,
                default_initializer=_get_default_param_initializer())
        else:
            self.filter_z = None

    def forward(self, x, z=None):
        if self._fused_add and z == None:
            raise ValueError("z can not be None")

        # intermediate_out for x
        bn_param_dtype = fluid.core.VarDesc.VarType.FP32
        bit_mask_dtype = fluid.core.VarDesc.VarType.INT32
        out = self._helper.create_variable_for_type_inference(x.dtype)
        bit_mask = self._helper.create_variable_for_type_inference(
            bit_mask_dtype)
        conv_x = self._helper.create_variable_for_type_inference(x.dtype)
        sum_x = self._helper.create_variable_for_type_inference(bn_param_dtype)
        sqsum_x = self._helper.create_variable_for_type_inference(
            bn_param_dtype)
        saved_mean_x = self._helper.create_variable_for_type_inference(
            bn_param_dtype)
        saved_invstd_x = self._helper.create_variable_for_type_inference(
            bn_param_dtype)
        running_mean_x = self._helper.create_variable_for_type_inference(
            bn_param_dtype)
        running_var_x = self._helper.create_variable_for_type_inference(
            bn_param_dtype)
        eq_scale_x = self._helper.create_variable_for_type_inference(x.dtype)
        eq_bias_x = self._helper.create_variable_for_type_inference(x.dtype)
        conv_z = self._helper.create_variable_for_type_inference(
            z.dtype) if self._has_shortcut else None
        sum_z = self._helper.create_variable_for_type_inference(
            bn_param_dtype) if self._has_shortcut else None
        sqsum_z = self._helper.create_variable_for_type_inference(
            bn_param_dtype) if self._has_shortcut else None
        saved_mean_z = self._helper.create_variable_for_type_inference(
            bn_param_dtype) if self._has_shortcut else None
        saved_invstd_z = self._helper.create_variable_for_type_inference(
            bn_param_dtype) if self._has_shortcut else None
        running_mean_z = self._helper.create_variable_for_type_inference(
            bn_param_dtype) if self._has_shortcut else None
        running_var_z = self._helper.create_variable_for_type_inference(
            bn_param_dtype) if self._has_shortcut else None
        eq_scale_z = self._helper.create_variable_for_type_inference(
            z.dtype) if self._has_shortcut else None
        eq_bias_z = self._helper.create_variable_for_type_inference(
            z.dtype) if self._has_shortcut else None
        if fluid.framework.in_dygraph_mode():
            out_list = _C_ops.resnet_unit(
                x, self.filter_x, z, self.filter_z, self._ele_count,
                self._stride, self._padding, self._dilation, self._groups,
                self._momentum, self._eps, self._fused_add, self._has_shortcut,
                self._act)
            out = out_list[0]
        else:
            inputs = {
                'X': x,
                'FilterX': self.filter_x,
                'Z': z,
                'FilterZ': self.filter_z
            }

            attrs = {
                'ele_count': self._ele_count,
                'stride': self._stride,
                'pad': self._padding,
                'dilate': self._dilation,
                'group': self._groups,
                'momentum': self._momentum,
                'epsilon': self._eps,
                'fused_add': self._fused_add,
                'has_shortcut': self._has_shortcut,
                'act': self._act
            }

            outputs = {
                'Y': out,
                'BitMask': bit_mask,
                'ConvX': conv_x,
                'SumX': sum_x,
                'SqSumX': sqsum_x,
                'SavedMeanX': saved_mean_x,
                'SavedInvstdX': saved_invstd_x,
                'RunningMeanX': running_mean_x,
                'RunningVarX': running_mean_z,
                'EqScaleX': eq_scale_x,
                'EqBiasX': eq_bias_x,
                'ConvZ': conv_z,
                'SumZ': sum_z,
                'SqSumZ': sqsum_z,
                'SavedMeanZ': saved_mean_z,
                'SavedInvstdZ': saved_invstd_z,
                'RunningMeanZ': running_mean_z,
                'RunningVarZ': running_var_z,
                'EqScaleZ': eq_scale_z,
                'EqBiasZ': eq_bias_z
            }

            self._helper.append_op(
                type="resnet_unit", inputs=inputs, outputs=outputs, attrs=attrs)

        return out
