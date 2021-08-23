// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/resnet/cudnn_resnet_unit.h"
#include "paddle/fluid/operators/resnet/cudnn_resnet_unit_impl.cu.h"

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
    Tensor *conv_out;
    Tensor *sum;
    Tensor *sum_of_squares;
    Tensor *tmp1;
    Tensor *tmp2;
    Tensor *tmp3;
    Tensor *tmp4;
    Tensor *tmp5;
    Tensor *tmp6;

    auto *output = ctx.Output<Tensor>("Y");
    // No implementations of this op exist with T = double, so output stats
    // pointers will always be float.
    T *conv_out_ptr = nullptr;
    float *sum_ptr = nullptr;
    float *sum_of_squares_ptr = nullptr;
    auto output_shape = framework::vectorize<int>(output->dims());
    int output_channel = output_shape.back();
    conv_out_ptr =
        conv_out->mutable_data<float>(output->dims(), output->place());
    sum_ptr = sum->mutable_data<float>(framework::make_ddim({output_channel}),
                                       output->place());
    sum_of_squares_ptr = sum_of_squares->mutable_data<float>(
        framework::make_ddim({output_channel}), output->place());

    // 1. Conv
    CuDNNNormConvolutionOp<T> conv_op = new CuDNNNormConvolutionOp<T>(true);
    conv_op.Init(ctx);
    conv_op.Forward(ctx, conv_out_ptr, sum_ptr, sum_of_squares_ptr);
    // 2. BN

    // 3. scale + bias + add + relu
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
