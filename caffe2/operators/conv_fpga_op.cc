#include "caffe2/operators/conv_fpga_op.h"
#include "caffe2/operators/conv_op_fpga_impl.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

const char kConvDoc[] = R"DOC(

Operator that calls the FPGA GEMM KERNEL to implement convolution operation and im2col on FPGA.

Added by jirui.

)DOC";

std::function<void(OpSchema&)> Conv_fpga_DocGenerator(const char* dim) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
The convolution operator consumes an input vector, a {dim}filter blob
and a bias blob and computes the output. {conv_doc})DOC";
    c10::ReplaceAll(doc, "{dim}", dim);
    c10::ReplaceAll(doc, "{conv_doc}", kConvDoc);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data blob, of shape $(N, C_{in}, H_{in}, W_{in})$, to be convolved with the kernels in the filter blob."
      );
    schema.Input(
        1,
        "filter",
        "The filter blob, of shape $(M, C_{in}, K_H, K_W)$, containing the filters to be convolved with the data."
      );
    schema.Input(
        2,
        "bias",
        "The bias blob, of length $M$, containing the biases for the convolution, one bias per filter."
      );
    schema.Output(
        0,
        "Y",
        "Output data blob, of shape $(N, C_{out}, H_{out}, W_{out})$, that contains the result of the convolution."
      );
  };
}
REGISTER_CPU_OPERATOR(Conv_fpga, Conv_fpga_Op<float, CPUContext>);

OPERATOR_SCHEMA(Conv_fpga)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase<CPUContext>::CostInferenceForConv))
    .FillUsing(Conv_fpga_DocGenerator(""))
    .InheritOnnxSchema();
//
// REGISTER_CPU_OPERATOR(Conv1D, ConvOp<float, CPUContext>);
//
// OPERATOR_SCHEMA(Conv1D)
//     .NumInputs(2, 3)
//     .NumOutputs(1)
//     .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
//     .FillUsing(ConvDocGenerator("1D "))
//     .InheritOnnxSchema("Conv");
//
// REGISTER_CPU_OPERATOR(Conv2D, ConvOp<float, CPUContext>);
//
OPERATOR_SCHEMA(Conv2D_fpga)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase<CPUContext>::CostInferenceForConv))
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .FillUsing(Conv_fpga_DocGenerator("2D_fpga "))
    .InheritOnnxSchema("Conv_fpga");

// REGISTER_CPU_OPERATOR(Conv3D, ConvOp<float, CPUContext>);
//
// OPERATOR_SCHEMA(Conv3D)
//     .NumInputs(2, 3)
//     .NumOutputs(1)
//     .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
//         ConvPoolOpBase<CPUContext>::CostInferenceForConv))
//     .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
//     .FillUsing(ConvDocGenerator("3D "))
//     .InheritOnnxSchema("Conv");

} // namespace caffe2
