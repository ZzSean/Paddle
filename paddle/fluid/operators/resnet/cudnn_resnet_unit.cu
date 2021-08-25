// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/operators/resnet/cudnn_bn_stats_finalize.cu.h"
#include "paddle/fluid/operators/resnet/cudnn_norm_conv.cu.h"
#include "paddle/fluid/operators/resnet/cudnn_resnet_unit.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;

template <typename T>
class ResNetUnitKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("It must use CUDAPlace."));

    // temp tensor for intermediate results
    // norm conv
    Tensor *conv_out, *sum, *sum_of_squares;
    // bn stats finalize
    Tensor *saved_mean, *save_invstd, *running_mean, *running_var, *equiv_scale,
        *equiv_bias;

    auto *output = ctx.Output<Tensor>("Y");
    // No implementations of this op exist with T = double, so output stats
    // pointers will always be float.
    // T *conv_out_ptr = nullptr;
    // float *sum_ptr = nullptr;
    // float *sum_of_squares_ptr = nullptr;
    // float *running_mean_ptr = nullptr;
    // float *running_var_ptr = nullptr;
    // float *saved_mean_ptr = nullptr;
    // float *saved_invstd_ptr = nullptr;
    // T *equiv_scale_ptr = nullptr;
    // T *equiv_bias_ptr = nullptr;
    auto output_shape = framework::vectorize<int>(output->dims());
    int output_channel = output_shape.back();
    auto param_shape = framework::make_ddim({output_channel});
    auto place = output->place();

#define MALLOC_AND_GET_PTR(TR, Dtype, Shape, Place) \
  Dtype *TR##_ptr = TR->mutable_data<Dtype>(Shape, Place);

    MALLOC_AND_GET_PTR(conv_out, T, output->dims(), place)
    MALLOC_AND_GET_PTR(sum, float, param_shape, place)
    MALLOC_AND_GET_PTR(sum_of_squares, float, param_shape, place)
    MALLOC_AND_GET_PTR(saved_mean, float, param_shape, place)
    MALLOC_AND_GET_PTR(saved_invstd, float, param_shape, place)
    MALLOC_AND_GET_PTR(running_mean, float, param_shape, place)
    MALLOC_AND_GET_PTR(running_var, float, param_shape, place)
    MALLOC_AND_GET_PTR(equiv_scale, T, param_shape, place)
    MALLOC_AND_GET_PTR(equiv_bias, T, param_shape, place)

    // 1. Conv
    CuDNNNormConvolutionOp<T> conv_op = new CuDNNNormConvolutionOp<T>(true);
    conv_op.Init(ctx);
    conv_op.Forward(ctx, conv_out_ptr, sum_ptr, sum_of_squares_ptr);
    // 2. BN
    CuDNNBNStatsFinalizeOp<T> bn_op = new CuDNNBNStatsFinalizeOp<T>();
    bn_op.Init(ctx);
    bn_op.Forward(ctx, sum_ptr, sum_of_squares_ptr, saved_mean_ptr,
                  saved_invstd_ptr, running_mean_ptr, running_var_ptr,
                  equiv_scale_ptr, equiv_bias_ptr);

    // 3. scale + bias + add + relu
    bool has_shortcut = ctx.Attr<bool>("has_shortcut");
    bool fused_add = ctx.Attr<bool>("fused_add");
    if (has_shortcut) {
      // 3.1 Conv for second input
      // 3.2 BN for second input
    }
    CuDNNScaleBiasAddReluOp<T> sbar_op = new CuDNNScaleBiasAddReluOp<T>();
    sbar_op.Init(ctx);
    sbar_op.Forward(ctx);
#undef MALLOC_AND_GET_PTR
  }
};

template <typename T>
class ResNetUnitGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("It must use CUDAPlace."));
  }
};

}  // namespace operators
}  // namespace paddle

#if CUDNN_VERSION >= 8000
namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    resnet_unit, ops::ResNetUnitKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    resnet_unit_grad,
    ops::ResNetUnitGradKernel<plat::CUDADeviceContext, plat::float16>);
#endif
