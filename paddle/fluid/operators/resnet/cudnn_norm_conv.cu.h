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

#include "paddle/fluid/operators/resnet/cudnn_fusion_helper.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using dynload = platform::dynload;
template <typename T>
class CuDNNNormConvolutionOp {
 public:
  explicit CuDNNNormConvolutionOp(bool first_input)
#if CUDNN_VERSION >= 8000
      : fwd_op_(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS),
        bwd_wgrad_op_(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD),
#endif
        first_input_(first_input) {
    PADDLE_ENFORCE_CUDA_SUCCESS(cudnnCreateTensorDescriptor(&in_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudnnCreateTensorDescriptor(&equiv_scale_bias_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(cudnnCreateTensorDescriptor(&out_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(cudnnCreateTensorDescriptor(&in_stats_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(cudnnCreateTensorDescriptor(&out_stats_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(cudnnCreateFilterDescriptor(&filter_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(cudnnCreateConvolutionDescriptor(&conv_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudnnCreateActivationDescriptor(&activation_desc_));
  }

  ~CuDNNNormConvolutionOp() {
    PADDLE_ENFORCE_CUDA_SUCCESS(cudnnDestroyTensorDescriptor(in_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudnnDestroyTensorDescriptor(equiv_scale_bias_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(cudnnDestroyTensorDescriptor(out_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(cudnnDestroyTensorDescriptor(in_stats_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(cudnnDestroyTensorDescriptor(out_stats_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(cudnnDestroyFilterDescriptor(filter_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(cudnnDestroyConvolutionDescriptor(conv_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudnnDestroyActivationDescriptor(activation_desc_));
  }

  void Init(const framework::ExecutionContext &ctx) {
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    auto cudnn_fwd_compute_type = platform::CudnnDataType<float>::type;
    dtype_ = platform::CudnnDataType<T>::type;
    format_ = CUDNN_TENSOR_NHWC;

    InitDescriptors(ctx);

    // Have cuDNN make a 'plan' for the fused op, returning the temp workspace
    // size required.
    GetTempSize(ctx);
#endif  // CUDNN_VERSION >= 7600
  }

  void Forward(const platform::ExecutionContext &ctx, T *output_ptr,
               float *sum_ptr, float *sum_of_squares_ptr) {
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();

    auto *input =
        first_input_ ? ctx.Input<Tensor>("X") : ctx.Input<Tensor>("Z");
    auto *filter = first_input_ ? ctx.Input<Tensor>("FilterX")
                                : ctx.Input<Tensor>("FilterZ");
    auto *scale = ctx.Input<Tensor>("Scale");
    auto *bias = ctx.Input<Tensor>("Bias");
    T *input_ptr = input->data<T>();
    T *filter_ptr = filter->data<T>();
    if (fwd_workspace_byte_ > workspace_handle::WorkspaceSize()) {
      workspace_handle.ReallocWorkspace(fwd_workspace_byte_);
    }
    auto workspace_ptr = workspace_handle.allocation_;

#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    T *equiv_scale_ptr = nullptr;
    T *equiv_bias_ptr = nullptr;
    if (fprop_eq_scale_bias_ptr_type_ != CUDNN_PTR_NULL) {
      equiv_scale_ptr = scale->data<T>();
      equiv_bias_ptr = bias->data<T>();
    }

    // This operator does not support output blending as specified by alpha or
    // beta.
    // Set data input pointers in op instance
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA, input_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WDATA, filter_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQSCALE, equiv_scale_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQBIAS, equiv_bias_ptr);
    // Set workspace input pointer in op instance
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE, workspace_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(
        CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &fwd_workspace_byte_);
    // Set data output pointers in op instance
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_YDATA, output_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_YSUM, sum_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_YSQSUM, sum_of_squares_ptr);
    // Launch forward operation
    fwd_op_.Execute(handle);
#endif  // CUDNN_VERSION < 7600
  }

  void Backward(const platform::ExecutionContext &ctx) {}

 private:
  void InitDescriptors(const platform::ExecutionContext &ctx) {
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    // We'll need normconv fprop if we're doing a normalization, activation, or
    // outputting stats.
    bool fused_normconv_fprop_needed = true;
    // We only supply a null pointer if we think cuDNN can then fall back to the
    // conventional fprop.
    fprop_eq_scale_bias_ptr_type_ =
        fused_normconv_fprop_needed ? CUDNN_PTR_16B_ALIGNED : CUDNN_PTR_NULL;

    // We'll need normconv wgrad if we're doing a normalization or activation.
    bool fused_normconv_wgrad_needed = false;
    // We only supply a null pointer if we think cuDNN can then fall back to the
    // conventional wgrad.
    wgrad_eq_scale_bias_ptr_type_ =
        fused_normconv_wgrad_needed ? CUDNN_PTR_16B_ALIGNED : CUDNN_PTR_NULL;
    auto stats_ptr_type = CUDNN_PTR_16B_ALIGNED;

    // Describe i/o tensor pointer alignment for forward fused op
    fwd_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_XDATA_PLACEHOLDER, CUDNN_PARAM_WDATA_PLACEHOLDER,
         CUDNN_PARAM_YDATA_PLACEHOLDER},
        CUDNN_PTR_16B_ALIGNED);
    fwd_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
        fprop_eq_scale_bias_ptr_type_);
    fwd_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_YSUM_PLACEHOLDER, CUDNN_PARAM_YSQSUM_PLACEHOLDER},
        stats_ptr_type);

    // Describe i/o tensor pointer alignment for backward wgrad fused op
    bwd_wgrad_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_DYDATA_PLACEHOLDER, CUDNN_PARAM_XDATA_PLACEHOLDER,
         CUDNN_PARAM_DWDATA_PLACEHOLDER},
        CUDNN_PTR_16B_ALIGNED);
    bwd_wgrad_op_.SetOpConstParamAttr(
        {CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
        wgrad_eq_scale_bias_ptr_type_);

    auto pad = ctx.Attr<int>("pad");
    auto stride = ctx.Attr<int>("stride");
    auto dilate = ctx.Attr<int>("dilate");
    auto group = ctx.Attr<int>("group");
    auto *filter = first_input_ ? ctx.Input<Tensor>("FilterX")
                                : ctx.Input<Tensor>("FilterZ");
    auto *input =
        first_input_ ? ctx.Input<Tensor>("X") : ctx.Input<Tensor>("Z");
    auto *output = ctx.Input<Tensor>("Y");
    auto filter_shape = filter->dims();
    auto input_shape = framework::vectorize<int>(input->dims());
    auto input_stride = GetStrides(input_shape);
    auto output_shape = framework::vectorize<int>(output->dims());
    auto output_stride = GetStrides(output_shape);
    // set conv desc
    PADDLE_ENFORCE_CUDA_SUCCESS(cudnnSetConvolution2dDescriptor(
        conv_desc_, pad, pad, stride, stride, dilate, dilate,
        CUDNN_CROSS_CORRELATION, cudnn_fwd_compute_type));
    cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH;
    // if (GetEnvAllowTensorCore() && GetEnvAllowTensorCoreConversion() &&
    //     (DataType<DType>::kFlag != kFloat16))
    //   math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnSetConvolutionMathType(conv_desc_, math_type));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnSetConvolutionGroupCount(conv_desc_, group));
    // set filter desc
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetFilter4dDescriptor(
        filter_desc_, dtype_, format_, filter_shape[0], filter_shape[1],
        filter_shape[2], filter_shape[3]));

    // set input desc
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
        in_desc_, dtype_, static_cast<int>(input_shape.size()),
        input_shape.data(), input_stride.data()));

    // set output desc
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
        out_desc_, dtype_, static_cast<int>(output_shape.size()),
        output_shape.data(), output_stride.data()));

    // set scale/bias descriptors
    int input_channel =
        format_ == CUDNN_TENSOR_NHWC ? input_shape.back() : input_shape[1];
    TShape equiv_scale_bias_shape = TShape({in_features});
    std::vector<int> equiv_scale_shape = {1, static_cast<int>(input_channel), 1,
                                          1};
    std::vector<int> equiv_scale_stride = {static_cast<int>(input_channel), 1,
                                           1, 1};
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
        equiv_scale_bias_desc_, dtype_,
        static_cast<int>(equiv_scale_shape.size()), equiv_scale_shape.data(),
        equiv_scale_stride.data()));
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                equiv_scale_bias_desc_);
    bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                      equiv_scale_bias_desc_);

    int output_channel =
        format_ == CUDNN_TENSOR_NHWC ? output_shape.back() : output_shape[1];
    std::vector<int> stats_shape = {1, output_channel, 1, 1};
    std::vector<int> stats_stride = {output_channel, 1, 1, 1};
    // Stats are output in the same precision as the forward compute (i.e.
    // float32)
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
        out_stats_desc_, cudnn_fwd_compute_type,
        static_cast<int>(stats_shape.size()), stats_shape.data(),
        stats_stride.data()));
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YSTATS_DESC, out_stats_desc_);

    // Here's where the standard convolution does a 'SelectAlgo', which may run
    // cudnnFind()
    // Not available yet for the NormConvolution operation.
    // If we're allowing Tensor Core variants of the algos to be considered in

    // Copied temporarily from 'SelectAlgo': probably not needed

    // *Find*() or *Get*(), but a non-Tensor-Core algo variant is the fastest,
    // we must change the descriptor to preclude Tensor Core.  Simplest is to
    // once again set the mathType in all cases.
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));

    // Set activation descriptor, default is no activation
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_IDENTITY;
    if (ctx.Attr<string>("act_type")) {
      PADDLE_ENFORCE_EQ(
          ctx.Attr<string>("act_type"), "relu",
          platform::errors::InvalidArgument(
              "Only relu activation supported in normalized convolution."));
      mode = CUDNN_ACTIVATION_RELU;
    }
    auto nan_prop = CUDNN_NOT_PROPAGATE_NAN;
    double dummy_clip = 0.0;
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetActivationDescriptor(
        activation_desc_, mode, nan_prop, dummy_clip));
    // Currently, the only way to turn off activation is to not set the
    // descriptor
    if (mode != CUDNN_ACTIVATION_IDENTITY) {
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_DESC,
                                  activation_desc_);
      bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_DESC,
                                        activation_desc_);
    }

    // Set desc pointers
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_XDESC, in_desc_);
    bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_XDESC, in_desc_);
    // FusedOp does not accept CUDNN_BATCHNORM_PER_ACTIVATION
    fwd_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE, CUDNN_BATCHNORM_SPATIAL);
    bwd_wgrad_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE,
                                      CUDNN_BATCHNORM_SPATIAL);

    // The Cudnn Convolution op provides parameters for controlling math
    // precision
    // separately for forward and backward, and so there are separate forward
    // and backward conv
    // descriptors.  However, NormConvolution does not have these extra
    // parameters, so the
    // same descriptor can be used for both.
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_CONV_DESC, conv_desc_);
    bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_CONV_DESC, conv_desc_);
    // W desc for forward == dW desc for backward wgrad
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_WDESC, filter_desc_);
    bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_DWDESC, filter_desc_);
    // Y desc for forward == dY desc for backward wgrad
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YDESC, out_desc_);
    bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_DYDESC, out_desc_);
#endif  // CUDNN_VERSION < 7600
  }

  void GetTempSize(const platform::ExecutionContext &ctx) {
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    // Make op plan for forward op and set forward workspace size
    fwd_workspace_byte_ = fwd_op_.GetWorkspaceSizeInBytes(handle);
    // Make op plan for backward wgrad op and set backward wgrad workspace size
    bwd_wgrad_workspace_byte_ = bwd_wgrad_op_.GetWorkspaceSizeInBytes(handle);
    // Get workspace for backward dgrad- convolution requirement
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, filter_desc_, out_desc_, conv_desc_, in_desc_,
            back_conv_dgrad_algo_, &bwd_dgrad_conv_workspace_byte_));
    // cudaMalloc returns addresses that are aligned for large accesses (e.g. to
    // 512 bytes).
    // Since we may make one allocation and divide it into two parts when we
    // parallelize
    // the dgrad and wgrad kernels, we round the size of the wgrad tempspace up
    // to this
    // alignment size so the temp space dptrs for the dgrad kernels will respect
    // this alignment
    // when stacked on top of the wgrad temp area.
    const size_t dptr_alignment = 512;
    bwd_wgrad_workspace_byte_ =
        AlignUp(bwd_wgrad_workspace_byte_, dptr_alignment);
#endif  // CUDNN_VERSION < 7600
  }

  // Temp workspace size in bytes needed for Forward() operation.
  size_t fwd_workspace_byte_;
  // Temp workspace size in bytes needed for Backward() wgrad operation.
  size_t bwd_wgrad_workspace_byte_;
  // Temp workspace size in bytes needed for Backward() dgrad operation (conv
  // portion).
  size_t bwd_dgrad_conv_workspace_byte_;
  // The hardwired backward dgrad convolution algo
  cudnnConvolutionBwdDataAlgo_t back_conv_dgrad_algo_ =
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  // Should dgrad and wgrad be launched into separate streams
  bool parallelize_backward_kernels_;

  cudnnDataType_t dtype_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t equiv_scale_bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnTensorDescriptor_t in_stats_desc_;
  cudnnTensorDescriptor_t out_stats_desc_;
  // Convolution descriptor for forward and backward operation (same math type
  // used in both)
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorFormat_t format_;
  // The assumption of the fwd_op plan as to whether sum and sum_of_squares
  // outputs are populated.
  bool fwd_op_plan_output_stats_ = true;
#if CUDNN_VERSION >= 8000
  // A cached copy of the fwd_op plan ptr placeholder for equiv_stats and
  // equiv_bias.
  cudnnFusedOpsPointerPlaceHolder_t fprop_eq_scale_bias_ptr_type_;
  // A cached copy of the bwd_op plan ptr placeholder for equiv_stats and
  // equiv_bias.
  cudnnFusedOpsPointerPlaceHolder_t wgrad_eq_scale_bias_ptr_type_;
#endif

  // Specifies activation parameters: relu
  cudnnActivationDescriptor_t activation_desc_;

  bool first_input_ = true;

#if CUDNN_VERSION >= 8000
  // New normalized convolution forward fused-op
  CuDNNFusionOp fwd_op_;
  // New normalized convolution backward wgrad fused-op
  CuDNNFusionOp bwd_wgrad_op_;
#endif
};
}  // namespace operators
}  // namespace paddle
