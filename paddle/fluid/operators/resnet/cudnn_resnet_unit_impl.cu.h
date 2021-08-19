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

#include "paddle/fluid/operators/resnet/cudnn_fusion_helper.h"

namespace paddle {
namespace operators {
template <typename T>
class CuDNNNormConvolutionOp {
 public:
  CuDNNNormConvolutionOp()
#if CUDNN_VERSION >= 8000
      : fwd_op_(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS),
        bwd_wgrad_op_(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD),
#endif
  {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&equiv_scale_bias_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_stats_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_stats_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_desc_));
  }

  ~CuDNNNormConvolutionOp() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(equiv_scale_bias_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_stats_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_stats_desc_));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc_));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activation_desc_));
  }

  void Init(const framework::ExecutionContext &ctx) {
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 8.0 or later is required.";
#else
    auto cudnn_fwd_compute_type = platform::CudnnDataType<float>::type;
    dtype_ = platform::CudnnDataType<T>::type;
    // For float16 input type beta, gamma, mean, and average are stored in
    // float32.
    // For other input types, these parameters have the same type as input
    dtype_param_;

    format_ = CUDNN_TENSOR_NHWC;

    // Double check to make sure this class supports the operation
    if (!Supports())
      LOG(FATAL) << "Unexpected unsupported use of NormConvolution op.";

    InitDescriptors(ctx);

    // Have cuDNN make a 'plan' for the fused op, returning the temp workspace
    // size required.
    GetTempSize(ctx);

    // Create an equivalent BatchNormParam for the held instance of the
    // NhwcBatchNormOp
    // Not needed for Backward
    bn_param_.eps = 0.0;
    // Not needed for Backward since running mean/var are updated by forward
    // kernel.
    bn_param_.momentum = 0.f;
    // Finalize kernel can respond to fix_gamma = true
    bn_param_.fix_gamma = false;
    // use_global_stats will only be true for inference-only graphs where
    // backward is not needed
    bn_param_.use_global_stats = false;
    // Should have no effect on NHWCBatchNorm::Backward()
    bn_param_.output_mean_var = true;
    // NormConvolution only supported for NHWC layouts
    CHECK_EQ(effective_layout, mshadow::kNHWC);
    bn_param_.axis = 3;
    // Only cudnn NormConvolution is implemented
    bn_param_.cudnn_off = false;
    // Copy act_type value from NormalizeConvolutionParam -> BatchNormParam
    if (param_.act_type.has_value()) bn_param_.act_type = param_.act_type;
    bn_param_.bn_group = 1;
    bn_param_.xbuf_ptr = 0U;

    if (!param_.no_norm) {
      int in_features = static_cast<int>(Features(in_shapes[norm_conv::kData]));
      FinalizeInit(param_, Shape1(in_features), ctx);
    }

#endif  // CUDNN_VERSION >= 7600
  }

  void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    size_t expected_inputs = norm_conv::NumInputs(param_.no_norm);
    size_t expected_outputs = norm_conv::NumOutputs(param_.no_norm);
    CHECK_EQ(in_data.size(), expected_inputs);
    CHECK_EQ(out_data.size(), expected_outputs);
    CHECK_EQ(req.size(), expected_outputs);
    // Finalize checks
    if (!param_.no_norm) {
      CHECK(req[norm_conv::kEquivScale] == kWriteTo ||
            req[norm_conv::kEquivScale] == kNullOp);
      CHECK(req[norm_conv::kEquivBias] == kWriteTo ||
            req[norm_conv::kEquivBias] == kNullOp);
    }
    // All inputs (except for the data and weight) should have
    // a shape equal to the one used to init the op
    for (size_t i = 0; i != in_data.size(); ++i) {
      if (i == static_cast<size_t>(norm_conv::kData) ||
          i == static_cast<size_t>(norm_conv::WeightIdx(param_.no_norm)))
        continue;
      CHECK_EQ(in_data[i].shape_, init_shape_);
    }
    // All outputs (except for the first 3) should have a shape equal to the one
    // used to init the op
    for (size_t i = 3; i != out_data.size(); ++i) {
      CHECK_EQ(out_data[i].shape_, init_shape_);
    }

    if (!param_.no_norm) FinalizeForward(ctx, in_data, req, out_data);

    Stream<gpu> *s = ctx.get_stream<gpu>();
    auto workspace =
        AllocateTempWorkspace(ctx, fwd_workspace_byte_, norm_conv::kTempSpace);
    size_t workspace_size = TensorSizeBytes(workspace);

    // I/O's should have 2 more dims than the kernel dim
    int weight_idx = norm_conv::WeightIdx(param_.no_norm);
    DType *data_ptr =
        GetNdPtr(in_data[norm_conv::kData], param_.kernel.ndim() + 2, s);
    DType *wmat_ptr =
        GetNdPtr(in_data[weight_idx], param_.kernel.ndim() + 2, s);

#if CUDNN_VERSION < 7600
    LOG(FATAL) << "cuDNN version 7.6 or later is required.";
#else
    int in_features =
        static_cast<int>(Features(in_data[norm_conv::kData].shape_));
    DType *equiv_scale_ptr = nullptr;
    DType *equiv_bias_ptr = nullptr;
    if (fprop_eq_scale_bias_ptr_type_ != CUDNN_PTR_NULL) {
      if (param_.no_norm) {
        equiv_scale_ptr = static_cast<DType *>(ones_feature_vector_hdl_.dptr);
        equiv_bias_ptr = static_cast<DType *>(zeros_feature_vector_hdl_.dptr);
      } else {
        equiv_scale_ptr =
            out_data[norm_conv::kEquivScale]
                .get_with_shape<gpu, 1, DType>(Shape1(in_features), s)
                .dptr_;
        equiv_bias_ptr =
            out_data[norm_conv::kEquivBias]
                .get_with_shape<gpu, 1, DType>(Shape1(in_features), s)
                .dptr_;
      }
    }

    DType *out_ptr =
        GetNdPtr(out_data[norm_conv::kOut], param_.kernel.ndim() + 2, s);
    // Make sure the op's present need for output_stats corresponds to the
    // assumed need at the time
    // the 'plan' was made.
    bool needed_output_stats = CuDNNNormConvolutionOp::OutputStats(ctx, req);
    CHECK_EQ(needed_output_stats, fwd_op_plan_output_stats_)
        << "Improper instance lookup for CuDNNNormConvolutionOp: improper "
           "'output_stats' bool.";

    int out_features =
        static_cast<int>(Features(out_data[norm_conv::kOut].shape_));
    // No implementations of this op exist with DType = double, so output stats
    // pointers
    // will always be float.
    float *sum_ptr = nullptr;
    float *sum_of_squares_ptr = nullptr;
    if (fwd_op_plan_output_stats_) {
      sum_ptr = out_data[norm_conv::kOutSum]
                    .get_with_shape<gpu, 1, float>(Shape1(out_features), s)
                    .dptr_;
      sum_of_squares_ptr =
          out_data[norm_conv::kOutSumOfSquares]
              .get_with_shape<gpu, 1, float>(Shape1(out_features), s)
              .dptr_;
    }

    CHECK_EQ(req[norm_conv::kOut], kWriteTo)
        << "In norm-conv output, expecting simple write of output, not add-to "
           "or inplace write.";

    // This operator does not support output blending as specified by alpha or
    // beta.
    // Set data input pointers in op instance
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA, data_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WDATA, wmat_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQSCALE, equiv_scale_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQBIAS, equiv_bias_ptr);
    // Set workspace input pointer in op instance
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE, workspace.dptr_);
    fwd_op_.SetOpVariantParamAttrPtr(
        CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &workspace_size);
    // Set data output pointers in op instance
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_YDATA, out_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_YSUM, sum_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_YSQSUM, sum_of_squares_ptr);
    // Launch forward operation
    fwd_op_.Execute(s->dnn_handle_);
#endif  // CUDNN_VERSION < 7600
  }

  void Backward(const OpContext &ctx, const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &fwd_in_data,
                const std::vector<TBlob> &fwd_out_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    size_t expected_inputs = norm_conv::NumInputs(param_.no_norm);
    CHECK_EQ(fwd_in_data.size(), expected_inputs);
    // We expect to see an in_grad tensor for all inputs,
    // except the moving_mean and moving_var 'aux inputs'.
    CHECK_EQ(in_grad.size(), norm_conv::NumNonAuxInputs(param_.no_norm));
    Stream<gpu> *s = ctx.get_stream<gpu>();
    // RAII object to handle syncing of the underlying auxiliary stream with the
    // primary stream
    SyncedGPUAuxStream s_wgrad = ctx.get_gpu_aux_stream();
    size_t dgrad_kernels_workspace_offset_byte =
        parallelize_backward_kernels_ ? bwd_wgrad_workspace_byte_ : 0;
    // The temp space bytes requested to cover the kernels called directly by
    // this routine (the
    // wgrad and conv-dgrag).  The nhwc_bn_op.Backward() will also request a
    // workspace to cover
    // the bn-dgrad (+ potentially the wgrad if parallelize_backward_kernels_ is
    // true).
    size_t backward_workspace_byte =
        parallelize_backward_kernels_
            ? bwd_wgrad_workspace_byte_ + bwd_dgrad_conv_workspace_byte_
            : std::max(bwd_wgrad_workspace_byte_,
                       bwd_dgrad_conv_workspace_byte_);
    auto workspace = AllocateTempWorkspace(ctx, backward_workspace_byte,
                                           norm_conv::kTempSpace);

    size_t workspace_size = TensorSizeBytes(workspace);

    // I/O's should have 2 more dims than the kernel dim
    int input_weight_idx = norm_conv::WeightIdx(param_.no_norm);
    int ingrad_weight_idx = norm_conv::BwdInGradIdx(input_weight_idx);
    // Ptr to the forward operation data input
    DType *data_ptr =
        GetNdPtr(fwd_in_data[norm_conv::kData], param_.kernel.ndim() + 2, s);
    // Ptr to the incoming gradient of the forward operation data output 'Y' (an
    // input here)
    DType *y_grad_ptr =
        GetNdPtr(out_grad[norm_conv::kOut], param_.kernel.ndim() + 2, s);
    // Ptr to the outgoing gradient of the forward operation weight input (an
    // output here)
    DType *wgt_grad_ptr =
        GetNdPtr(in_grad[ingrad_weight_idx], param_.kernel.ndim() + 2, s);

#if CUDNN_VERSION < 7600
    LOG(FATAL) << "cuDNN version 7.6 or later is required.";
#else
    int in_features =
        static_cast<int>(Features(fwd_in_data[norm_conv::kData].shape_));
    DType *equiv_scale_ptr = nullptr;
    DType *equiv_bias_ptr = nullptr;
    if (wgrad_eq_scale_bias_ptr_type_ != CUDNN_PTR_NULL) {
      if (param_.no_norm) {
        equiv_scale_ptr = static_cast<DType *>(ones_feature_vector_hdl_.dptr);
        equiv_bias_ptr = static_cast<DType *>(zeros_feature_vector_hdl_.dptr);
      } else {
        equiv_scale_ptr =
            fwd_out_data[norm_conv::kBwdEquivScale]
                .get_with_shape<gpu, 1, DType>(Shape1(in_features), s)
                .dptr_;
        equiv_bias_ptr =
            fwd_out_data[norm_conv::kBwdEquivBias]
                .get_with_shape<gpu, 1, DType>(Shape1(in_features), s)
                .dptr_;
      }
    }
    // WGRAD

    // This operator does not support output blending as specified by alpha or
    // beta.
    // Set data input pointers in op instance
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA, data_ptr);
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_DYDATA, y_grad_ptr);
    // Here we supply equiv_scale and equiv_bias ptrs, though perhaps
    // statically-initted ones
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQSCALE,
                                           equiv_scale_ptr);
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQBIAS, equiv_bias_ptr);
    // Set workspace input pointer in op instance
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE,
                                           workspace.dptr_);
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(
        CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &workspace_size);
    // Set data output pointers in op instance
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_DWDATA, wgt_grad_ptr);
    // Launch backward wgrad operation into the alternate stream (if enabled)
    bwd_wgrad_op_.Execute(s_wgrad.GetStream()->dnn_handle_);

    // DGRAD - convolution dgrad followed optionally by batchnorm dgrad

    if (req[norm_conv::kData] != kNullOp) {
      // First: convolution dgrad
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType beta = 0.0f;
      typename DataType<DType>::ScaleType beta_add = 1.0f;
      bool only_conv_dgrad_present =
          param_.no_norm && !param_.act_type.has_value();
      typename DataType<DType>::ScaleType conv_dgrad_beta =
          (only_conv_dgrad_present && (req[norm_conv::kData] == kAddTo))
              ? beta_add
              : beta;
      if (!only_conv_dgrad_present && (req[norm_conv::kData] == kAddTo))
        LOG(FATAL) << "NormConvolution dgrad output summation not supported "
                      "when applying stats (needs extra conv dgrad buffer not "
                      "yet allocated).";

      // Ptr to the forward operation weight input
      DType *wmat_ptr =
          GetNdPtr(fwd_in_data[input_weight_idx], param_.kernel.ndim() + 2, s);
      // Ptr to the outgoing gradient of the forward operation data input (an
      // output here)
      DType *x_grad_ptr =
          GetNdPtr(in_grad[norm_conv::kData], param_.kernel.ndim() + 2, s);
      auto *dgrad_workspace_ptr = reinterpret_cast<char *>(workspace.dptr_) +
                                  dgrad_kernels_workspace_offset_byte;
      size_t dgrad_workspace_byte =
          workspace_size - dgrad_kernels_workspace_offset_byte;
      if (!dgrad_as_gemm_) {
        // Launch conv dgrad into the primary stream, although with an offsetted
        // workspace
        // pointer if dual-stream is enabled.
        CUDNN_CALL(cudnnConvolutionBackwardData(
            s->dnn_handle_, &alpha, filter_desc_, wmat_ptr, out_desc_,
            y_grad_ptr, conv_desc_, back_conv_dgrad_algo_, dgrad_workspace_ptr,
            dgrad_workspace_byte, &conv_dgrad_beta, in_desc_, x_grad_ptr));
      } else {
        int in_features =
            static_cast<int>(Features(fwd_in_data[norm_conv::kData].shape_));
        int GEMM_M = in_features;
        int GEMM_N =
            static_cast<int>(GetNHW(fwd_in_data[norm_conv::kData].shape_));
        int output_features =
            static_cast<int>(Features(out_grad[norm_conv::kOut].shape_));
        int GEMM_K = output_features;

        int lda = in_features;
        int ldb = output_features;
        int ldc = in_features;

        if (dgrad_as_gemm_debug_) {
          LOG(INFO) << "Using DGRAD AS GEMM for " << GEMM_M << "x" << GEMM_N
                    << "x" << GEMM_K;
        }

        mshadow::Tensor<gpu, 2, DType> wmat =
            fwd_in_data[input_weight_idx].get_with_shape<gpu, 2, DType>(
                Shape2(GEMM_M, GEMM_K), s);
        mshadow::Tensor<gpu, 2, DType> grad =
            out_grad[norm_conv::kOut].get_with_shape<gpu, 2, DType>(
                Shape2(GEMM_K, GEMM_N), s);
        mshadow::Tensor<gpu, 2, DType> gdata =
            in_grad[norm_conv::kData].get_with_shape<gpu, 2, DType>(
                Shape2(GEMM_M, GEMM_N), s);

        int32_t algo_precision =
            -1;  // No params to set policy for this, so pseudo-fp16.
        CuBLASFullyConnectedOp<DType>::CublasGemm(
            s, false, false, GEMM_M, GEMM_N, GEMM_K, req[norm_conv::kData],
            wmat, grad, gdata, lda, ldb, ldc, algo_precision,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }

      // Second (if needed): batchnorm dgrad
      if (!only_conv_dgrad_present) {
        // The following unusual case is not typically found (e.g. in Resnet).
        if (param_.no_norm)
          LOG(FATAL) << "Relu activation with no_norm not yet supported.";
        // Prepare inputs of NHWCBatchnorm::Backward()
        // Note that the 1st input is the same as the 1st output, i.e. the
        // Batchnorm
        // is operating 'in place' on the gradient as output by the convolution
        // dgrad.
        TBlob not_used;
        std::vector<TBlob> bn_bwd_inputs{
            in_grad[norm_conv::kData],
            fwd_out_data[norm_conv::kBwdSavedMean],
            fwd_out_data[norm_conv::kBwdSavedInvStdDev],
            fwd_in_data[norm_conv::kData],
            fwd_in_data[norm_conv::kGamma],
            fwd_in_data[norm_conv::kBeta],
            not_used,
            not_used};
        std::vector<OpReqType> bn_bwd_req{
            req[norm_conv::kData],
            req[norm_conv::BwdInGradIdx(norm_conv::kGamma)],
            req[norm_conv::BwdInGradIdx(norm_conv::kBeta)]};
        std::vector<TBlob> bn_bwd_outputs{
            in_grad[norm_conv::kData],
            in_grad[norm_conv::BwdInGradIdx(norm_conv::kGamma)],
            in_grad[norm_conv::BwdInGradIdx(norm_conv::kBeta)]};
        // This function will ask for a temp workspace and will get the same
        // pointer
        // (as long as the workspace does not to be increased).  This all works
        // fine because
        // at this point the wgrad, conv-dgad and this bn-dgrad are all using
        // the same stream.

        // The Init call is made prior to each Backward(), a historical result
        // of transitioning
        // from a symbolic to a gluon (imperative) op style.
        nhwc_bn_op.Init(bn_param_);
        // Launch batchnorm backward into the primary stream.  This will launch
        // a kernel with
        // an offsetted workspace pointer if dual-stream is enabled.
        nhwc_bn_op.Backward(ctx, bn_bwd_inputs, bn_bwd_req, bn_bwd_outputs,
                            dgrad_kernels_workspace_offset_byte);
      }
    }
#endif  // CUDNN_VERSION < 7600
  }

  /*!
   * \brief Returns whether the norm convolution operation described by `param`
   * is supported.
   */
  template <typename SupportedConvParam>
  static bool Supports(const SupportedConvParam &param,
                       const TShape &in_data_shape, int dev_id) {
    using namespace mshadow;
    static_assert(
        std::is_same<SupportedConvParam, ConvolutionParam>::value ||
            std::is_same<SupportedConvParam, NormConvolutionParam>::value,
        "Unsupported template specialization of NormConvolution::Supports()");
    // Need cuDNN version >= 7.6
    if (CUDNN_VERSION < 7600) return false;
    // Volta (70), Turing (75) and Ampere (80, 86) GPU architectures supported.
    static const std::set<int> supported_arches{70, 75, 80, 86};
    if (supported_arches.count(SMArch(dev_id)) == 0) return false;
    // On Ampere bn(a) + conv is not supported before the cuDNN 8.0.4 release
    if (CUDNN_VERSION < 8004 && SMArch(dev_id) >= 80 &&
        !dmlc::GetEnv("MXNET_EXTENDED_NORMCONV_SUPPORT", false))
      return false;
    // Only kNHWC and kNWC format supported
    auto layout_val = param.layout.value();
    if (layout_val != kNWC && layout_val != kNHWC) return false;
    // Only 2D convolution supported
    if (param.kernel.ndim() != 2) return false;
    // Only 3x3 and 1x1 convolution supported
    if (SMArch(dev_id) == 70) {
      if (!(param.kernel == TShape{3, 3}) &&
          !(param.kernel == TShape{1, 1} && param.stride == TShape{1, 1}))
        return false;
    } else {
      if (!(param.kernel == TShape{3, 3}) && !(param.kernel == TShape{1, 1}))
        return false;
    }
    // No dilation supported
    if (param.dilate != TShape{1, 1}) return false;
    // No grouped convolution supported
    if (param.num_group != 1) return false;
    // Must have a multiple of 32 input features 'c' (assumes N..C layout).
    if (in_data_shape[in_data_shape.ndim() - 1] % 32 != 0) return false;
    // Must have a multiple of 32 output features (== number of filters 'k')
    if (param.num_filter % 32 != 0) return false;
    // Op parameters are supported, assuming datatype is float16
    return DataType<DType>::kFlag == kFloat16;
  }

 private:
  void InitDescriptors(const ExecutionContext &ctx, bool first_input) {
#if CUDNN_VERSION < 8000
    LOG(FATAL) << "cuDNN version 7.6 or later is required.";
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
    auto *filter = first_input ? ctx.Input<Tensor>("FilterX")
                               : ctx.Input<Tensor>("FilterZ");
    auto filter_shape = filter->dims();
    // set conv desc
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc_, pad, pad, stride, stride, dilate, dilate,
        CUDNN_CROSS_CORRELATION, cudnn_fwd_compute_type));
    // set filter desc
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc_, dtype_, format_,
                                          filter_shape[0], filter_shape[1],
                                          filter_shape[2], filter_shape[3]));

    cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH;
    if (GetEnvAllowTensorCore() && GetEnvAllowTensorCoreConversion() &&
        (DataType<DType>::kFlag != kFloat16))
      math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
    CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc_, math_type));
    CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc_, param_.num_group));

    // set input desc
    std::vector<int> dshape_buffer(dshape.ndim());
    nnvm::ShapeTypeCast(dshape.begin(), dshape.end(), dshape_buffer.data());
    std::vector<int> dstride_buffer(dstride.ndim());
    nnvm::ShapeTypeCast(dstride.begin(), dstride.end(), dstride_buffer.data());

    CUDNN_CALL(cudnnSetTensorNdDescriptor(
        in_desc_, dtype_, static_cast<int>(dshape.ndim()), dshape_buffer.data(),
        dstride_buffer.data()));

    // set output desc
    std::vector<int> oshape_buffer(oshape.ndim());
    nnvm::ShapeTypeCast(oshape.begin(), oshape.end(), oshape_buffer.data());
    std::vector<int> ostride_buffer(ostride.ndim());
    nnvm::ShapeTypeCast(ostride.begin(), ostride.end(), ostride_buffer.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(
        out_desc_, dtype_, static_cast<int>(oshape.ndim()),
        oshape_buffer.data(), ostride_buffer.data()));

    // set scale/bias descriptors
    int in_features = static_cast<int>(Features(in_shapes[norm_conv::kData]));
    TShape equiv_scale_bias_shape = TShape({in_features});
    std::vector<int> equiv_scale_shape = {1, static_cast<int>(in_features), 1,
                                          1};
    std::vector<int> equiv_scale_stride = {static_cast<int>(in_features), 1, 1,
                                           1};
    CUDNN_CALL(cudnnSetTensorNdDescriptor(
        equiv_scale_bias_desc_, dtype_,
        static_cast<int>(equiv_scale_shape.size()), &equiv_scale_shape[0],
        &equiv_scale_stride[0]));
    if (SMArch(ctx.run_ctx.ctx.dev_id) == 70) {
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                  equiv_scale_bias_desc_);
      bool use_bn_wgrad = dmlc::GetEnv("MXNET_WGARD_NORMCONV_SUPPORT", true);
      if (use_bn_wgrad)
        bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                          equiv_scale_bias_desc_);
    } else {
      if (!param_.no_norm) {
        fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                    equiv_scale_bias_desc_);
        bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
                                          equiv_scale_bias_desc_);
      }
    }

    if (!param_.no_norm) {
      CHECK_EQ(equiv_scale_bias_shape, in_shapes[norm_conv::kInSum])
          << "Expecting equal equivalent-scale and sum input tensor shapes.";
      CHECK_EQ(equiv_scale_bias_shape, in_shapes[norm_conv::kInSumOfSquares])
          << "Expecting equal equivalent-scale and sum_squares input tensor "
             "shapes.";
      CHECK_EQ(equiv_scale_bias_shape, in_shapes[norm_conv::kMovingMean])
          << "Expecting equal equivalent-scale and saved-mean input tensor "
             "shapes.";
      CHECK_EQ(equiv_scale_bias_shape, in_shapes[norm_conv::kMovingVar])
          << "Expecting equal equivalent-scale and saved-inv-stddev input "
             "tensor shapes.";
      CHECK_EQ(equiv_scale_bias_shape, in_shapes[norm_conv::kGamma])
          << "Expecting equal equivalent-scale and gamma input tensor shapes.";
      CHECK_EQ(equiv_scale_bias_shape, in_shapes[norm_conv::kBeta])
          << "Expecting equal equivalent-scale and beta input tensor shapes.";
    }

    if (output_stats) {
      // Replace with checks at every Forward()?
      //      TShape sum_shape = out_shape[norm_conv::kOutSum];
      //      TShape sum_of_squares_shape =
      //      out_shape[norm_conv::kOutSumOfSquares];
      //      CHECK_EQ(sum_shape, sum_of_squares_shape) <<
      //        "Expecting equal sum and sum_of_squares output tensor shapes.";
      int output_features = static_cast<int>(Features(out_shape));
      std::vector<int> stats_shape = {1, output_features, 1, 1};
      std::vector<int> stats_stride = {output_features, 1, 1, 1};
      // Stats are output in the same precision as the forward compute (i.e.
      // float32)
      CUDNN_CALL(
          cudnnSetTensorNdDescriptor(out_stats_desc_, cudnn_fwd_compute_type,
                                     static_cast<int>(stats_shape.size()),
                                     &stats_shape[0], &stats_stride[0]));
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YSTATS_DESC, out_stats_desc_);
    }

    // Here's where the standard convolution does a 'SelectAlgo', which may run
    // cudnnFind()
    // Not available yet for the NormConvolution operation.
    // If we're allowing Tensor Core variants of the algos to be considered in

    // Copied temporarily from 'SelectAlgo': probably not needed

    // *Find*() or *Get*(), but a non-Tensor-Core algo variant is the fastest,
    // we must change the descriptor to preclude Tensor Core.  Simplest is to
    // once again set the mathType in all cases.
    CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));

    // Set activation descriptor, default is no activation
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_IDENTITY;
    if (param_.act_type.has_value()) {
      CHECK_EQ(param_.act_type.value(), activation::kReLU)
          << "Only relu activation supported in normalized convolution.";
      mode = CUDNN_ACTIVATION_RELU;
    }
    auto nan_prop = CUDNN_NOT_PROPAGATE_NAN;
    double dummy_clip = 0.0;
    CUDNN_CALL(cudnnSetActivationDescriptor(activation_desc_, mode, nan_prop,
                                            dummy_clip));
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

    // Set up 0.f and 1.f constant vectors if we're running in no-apply mode
    if ((fprop_eq_scale_bias_ptr_type_ != CUDNN_PTR_NULL ||
         wgrad_eq_scale_bias_ptr_type_ != CUDNN_PTR_NULL) &&
        param_.no_norm && !init_feature_vector_constants_) {
      size_t equiv_scale_bytes = in_features * sizeof(DType);
      Stream<gpu> *s = ctx.get_stream<gpu>();
      zeros_feature_vector_hdl_ =
          mxnet::Storage::Get()->Alloc(equiv_scale_bytes, Context::GPU());
      // Zero the read-only zeros_feature_vector_hdl_ area once.
      CUDA_CALL(cudaMemsetAsync(zeros_feature_vector_hdl_.dptr, 0,
                                equiv_scale_bytes,
                                mshadow::Stream<gpu>::GetStream(s)));
      ones_feature_vector_hdl_ =
          mxnet::Storage::Get()->Alloc(equiv_scale_bytes, Context::GPU());
      // Setting this up as 1's is a little tricky.  Not sure if the
      // cuMemsetD32Async would
      // have endian issues.
      TBlob ones_tblob(ones_feature_vector_hdl_.dptr, equiv_scale_bias_shape,
                       gpu::kDevMask, DataType<DType>::kFlag,
                       ctx.run_ctx.ctx.dev_id);
      auto ones_tensor =
          ones_tblob.get_with_shape<gpu, 1, DType>(Shape1(in_features), s);
      // Now init the ones tensor
      ones_tensor = 1.f;
      init_feature_vector_constants_ = true;
    }
#endif  // CUDNN_VERSION < 7600
  }

  void GetTempSize(const OpContext &ctx) {
#if CUDNN_VERSION < 7600
    LOG(FATAL) << "cuDNN version 7.6 or later is required.";
#else
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    // Make op plan for forward op and set forward workspace size
    fwd_workspace_byte_ = fwd_op_.GetWorkspaceSizeInBytes(s->dnn_handle_);
    // Make op plan for backward wgrad op and set backward wgrad workspace size
    bwd_wgrad_workspace_byte_ =
        bwd_wgrad_op_.GetWorkspaceSizeInBytes(s->dnn_handle_);
    // Get workspace for backward dgrad- convolution requirement
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
        s->dnn_handle_, filter_desc_, out_desc_, conv_desc_, in_desc_,
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
        RoundToMultiple(bwd_wgrad_workspace_byte_, dptr_alignment);
#endif  // CUDNN_VERSION < 7600
  }

  // Converts a TBlob to a dptr, checking for the expected dim and that it's
  // contiguous.
  DType *GetNdPtr(const TBlob &tb, int dim, Stream<gpu> *s) {
    DType *data_ptr = NULL;
    if (dim == 3) {
      Tensor<gpu, 3, DType> data = tb.get<gpu, 3, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 4) {
      Tensor<gpu, 4, DType> data = tb.get<gpu, 4, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 5) {
      Tensor<gpu, 5, DType> data = tb.get<gpu, 5, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else {
      LOG(FATAL) << "Unexpected Tensor size " << dim
                 << ", supporting only 3, 4 or 5.";
    }
    return data_ptr;
  }

  // Converts a TShape to a Shape<> of strides.
  // e.g. {shape[0], shape[1], shape[2]} -> {shape[1]*shape[2], shape[2], 1}
  template <int dim>
  inline Shape<dim> Strides(const TShape &s) {
    uint32_t ndim = s.ndim();
    TShape strides(ndim, -1);
    for (uint32_t i = 0; i != ndim; ++i) strides[i] = s.ProdShape(i + 1, ndim);
    return strides.get<dim>();
  }

  // Given a tensor shape of this operation, return the number of features 'c'
  int64_t Features(const TShape &dshape) {
    int c = 0;
    switch (dshape.ndim()) {
      case 3:
        c = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW)[1];
        break;
      case 4:
        c = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW)[1];
        break;
      case 5:
        c = ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW)[1];
        break;
      default:
        LOG(FATAL) << "Unexpected convolution data dimension " << dshape.ndim();
    }
    return c;
  }

  // Give a tensor shape of this operation, return the N * H * W
  int64_t GetNHW(const TShape &dshape) {
    int nhw = 0;
    switch (dshape.ndim()) {
      case 3: {
        auto tmp = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW);
        nhw = tmp[0] * tmp[2];
        break;
      }
      case 4: {
        auto tmp = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW);
        nhw = tmp[0] * tmp[2] * tmp[3];
        break;
      }
      case 5: {
        auto tmp =
            ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW);
        nhw = tmp[0] * tmp[3] * tmp[4];
        break;
      }
      default:
        LOG(FATAL) << "Unexpected convolution data dimension " << dshape.ndim();
    }
    return nhw;
  }
  std::vector<int> param_stride_;
  std::vector<int> param_dilate_;
  std::vector<int> param_pad_;

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
  NormConvolutionParam param_;
  // The assumption of the fwd_op plan as to whether sum and sum_of_squares
  // outputs are populated.
  bool fwd_op_plan_output_stats_;
#if CUDNN_VERSION >= 7600
  // A cached copy of the fwd_op plan ptr placeholder for equiv_stats and
  // equiv_bias.
  cudnnFusedOpsPointerPlaceHolder_t fprop_eq_scale_bias_ptr_type_;
  // A cached copy of the bwd_op plan ptr placeholder for equiv_stats and
  // equiv_bias.
  cudnnFusedOpsPointerPlaceHolder_t wgrad_eq_scale_bias_ptr_type_;
#endif

  // An instance of the equivalent Batchnorm operation, suitable for calling
  // Backward() on.
  NhwcBatchNormOp<DType> nhwc_bn_op;
  // The BatchNormParam associated with the NHWCBatchNormOp instance
  BatchNormParam bn_param_;

  bool init_feature_vector_constants_ = false;
  mxnet::Storage::Handle zeros_feature_vector_hdl_;
  mxnet::Storage::Handle ones_feature_vector_hdl_;

  // 1x1 conv dgrad as gemm
  bool dgrad_as_gemm_;
  bool dgrad_as_gemm_debug_;

  // Specifies activation parameters: relu
  cudnnActivationDescriptor_t activation_desc_;

  // data members to support finalize function
  int dtype_param_;
  // The shape used to init the in_stats_desc of the finalize op
  TShape init_shape_;

#if CUDNN_VERSION >= 8000
  // New normalized convolution forward fused-op
  CuDNNFusionOp fwd_op_;
  // New normalized convolution backward wgrad fused-op
  CuDNNFusionOp bwd_wgrad_op_;
#endif
};
}  // namespace operators
}  // namespace paddle