/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/resnet/cudnn_resnet_unit.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

void ResNetUnitOp::InferShape(framework::InferShapeContext *ctx) const {
  // check input
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ResNetUnitOp");
  OP_INOUT_CHECK(ctx->HasInput("FilterX"), "Input", "FilterX", "ResNetUnitOp");

  // check output
  OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "ResNetUnitOp");

  // TODO(zhangzheng): check dims for input and output
  const auto x_dims = ctx->GetInputDim("X");
  PADDLE_ENFORCE_GE(x_dims.size(), 2, platform::errors::InvalidArgument(
                                          "ShapeError: the dimensions of input "
                                          "must greater than or equal to 2."
                                          "But received: the shape of input "
                                          "= [%s], the dimension of input = "
                                          "[%d]",
                                          x_dims, x_dims.size()));
  PADDLE_ENFORCE_LE(x_dims.size(), 5, platform::errors::InvalidArgument(
                                          "ShapeError: the dimensions of input "
                                          "must smaller than or equal to 5."
                                          "But received: the shape of input "
                                          "= [%s], the dimension of input = "
                                          "[%d]",
                                          x_dims, x_dims.size()));
  // TODO(zhangzheng): infer shape of y
  // ctx->SetOutputDim("Y", y_dims);
}

framework::OpKernelType ResNetUnitOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");

  PADDLE_ENFORCE_EQ(input_data_type, ctx.Input<Tensor>("Scale")->type(),
                    platform::errors::InvalidArgument(
                        "The data type of Scale shoule be same as input type"));
  PADDLE_ENFORCE_EQ(input_data_type, ctx.Input<Tensor>("Bias")->type(),
                    platform::errors::InvalidArgument(
                        "The data type of Bias shoule be same as input type"));

  framework::LibraryType library = framework::LibraryType::kPlain;
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;

  return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                 library);
}

void ResNetUnitOpMaker::Make() {
  AddInput("X", "The input 1 tensor");
  AddInput("FilterX", "The filter tensor of input 1");
  AddInput("Z", "The input 2 tensor");
  AddInput("FilterZ", "The filter tensor of input 2");
  AddInput("Scale",
           "Scale is a 1-dimensional tensor of size C "
           "that is applied to the input before conv, always be 1 now");
  AddInput("Bias",
           "Bias is a 1-dimensional tensor of size C "
           "that is applied to the input before conv, always be 0 now");
  AddOutput("Y", "result after normalization");
  AddAttr<int>("elem_count", "");
  AddAttr<int>("stride", "").SetDefault(1);
  AddAttr<int>("pad", "").SetDefault(0);
  AddAttr<int>("dilate", "").SetDefault(1);
  AddAttr<float>("momentum", "").SetDefault(0.9);
  AddAttr<float>("epsilon", "").SetDefault(1e-5);
  AddAttr<bool>("fused_add", "").SetDefault(false);
  AddAttr<bool>("has_shorcut", "").SetDefault(false);
  AddAttr<std::string>("act_type", "The activation type to be fused.")
      .SetDefault("relu");
  AddComment(R"DOC(
****TODO****.
)DOC");
}

void ResNetUnitGradOp::InferShape(framework::InferShapeContext *ctx) const {
  // check input

  // check output
}

framework::OpKernelType ResNetUnitGradOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  const auto *var = ctx.InputVar(framework::GradVarName("Y"));
  if (var == nullptr) {
    PADDLE_THROW(platform::errors::NotFound(
        "Can not find Y@GRAD in the execution context."));
  }

  framework::LibraryType library = framework::LibraryType::kPlain;
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;

  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace(), layout,
      library);
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(resnet_unit, ops::ResNetUnitOp, ops::ResNetUnitOpMaker,
                  ops::ResNetUnitOpInferVarType,
                  ops::ResNetUnitGradOpMaker<paddle::framework::OpDesc>,
                  ops::ResNetUnitGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(resnet_unit_grad, ops::ResNetUnitGradOp);
