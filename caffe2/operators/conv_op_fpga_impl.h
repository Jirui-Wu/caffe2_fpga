// conv_op_fpga_impl.h is the templated implementation of the conv_op.h file.
#ifndef CAFFE2_OPERATORS_CONV_OP_FPGA_IMPL_H_
#define CAFFE2_OPERATORS_CONV_OP_FPGA_IMPL_H_
//FROM conv_op_fpga_impl
#include "caffe2/operators/conv_fpga_op.h"

#include <array>
#include <vector>
#include <cstdlib>
#include "caffe2/core/context.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"


//FROM host.hpp
#define PY_SSIZE_T_CLEAN
#include "caffe2/fpga/mesh_processor.hpp"
#include "caffe2/fpga/data_proc.hpp"
#include "caffe2/fpga/utils.hpp"
#include "caffe2/fpga/xcl2.hpp"
//#include <vector>
#define DATA_SIZE 4096
#include <unistd.h>
#include <assert.h>
#include <memory>
#include <random>
#include <typeinfo>
#include <string>
#include <iomanip>

//added
#include <fstream>
#include <iostream>
#include "caffe2/fpga/mm_fpga.hpp"

namespace caffe2 {

union cast_t {
    public:
        float f;
        unsigned i;
};

void FPGAGEMM (
    const int trans_A,
    const int trans_B,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C
  ){
        double fpga_times[3];
        math::Kernel(
         trans_A,
         trans_B,
         M,
         N,
         K,
         alpha,
         A,
         B,
         beta,
         C,
         fpga_times
       );


      #ifdef TIMER_ENABLED
       //TIMER ENEABLED MACRO CAN BE TURNED ON/OFF IN fpga/mesh_processor.hpp
       std::ofstream outfile("dataflow_test/timing.csv");
       if(!outfile.is_open()){
         std::cout<<"cannot open output file"<<std::endl;
         exit(EXIT_FAILURE);
       }
       else{
         for(int j=0; j<3; ++j){
           outfile <<"fpga_time "<<j<<" is "<<fpga_times[j]<<std::endl;
         }
       }
       outfile.close();
      #endif
     }

template <typename T, class Context>
bool Conv_fpga_Op<T, Context>::RunOnDeviceWithOrderNCHW(){
    //std::string modelName = "lenet";
	  std::cout << "Operations starting here" << std::endl;
    std::cout << "Loading inputs and weights ..." << std::endl;
    //std::string fileName = "data/" + modelName + "_imagenet/";

    const auto& X = Input(INPUT);
    const auto& filter = Input(FILTER);
    auto* Y = Output(0);
    const int N = X.dim32(0);
    const int C = X.dim32(1);
    const int G = group_;
    CAFFE_ENFORCE_EQ(X.dim(), filter.dim());
    const int M = filter.dim32(0);
    CAFFE_ENFORCE_EQ(
        C,
        filter.dim32(1) * G,
        "Convolution op: input channels does not match: # of input channels ",
        C,
        " is not equal to kernel channels * group: ",
        filter.dim32(1),
        "*",
        G);
    CAFFE_ENFORCE_EQ(
        M % G, 0, "The number of output channels is not divisible by group.");

    int kernel_size = 1;
    for (std::size_t i = 0; i < kernel_.size(); ++i) {
      CAFFE_ENFORCE_EQ(filter.dim32(i + 2), kernel_[i]);
      kernel_size *= kernel_[i];
    }
    ConvPoolOpBase<Context>::SetOutputSize(X, Y, M);

    if (N == 0) {
      Y->template mutable_data<T>();
      return true;
    }

    const vector<int> X_dims = GetDims(X);
    const vector<int> Y_dims = GetDims(*Y);
    const int X_HxW = X.numel() / (N * C);
    const int Y_HxW = Y->numel() / (N * M);
    const vector<int> img_shape(X.sizes().cbegin() + 1, X.sizes().cend());
    vector<int> buffer_shape(Y_dims.size() + 1);
    buffer_shape[0] = C * kernel_size;
    std::copy(Y_dims.cbegin(), Y_dims.cend(), buffer_shape.begin() + 1);

    const int buffer_size = C * kernel_size * Y_HxW;

    // The dimension of each kernel
    const int kernel_dim = C / G * kernel_size;
    const int X_stride = C * X_HxW;
    const int Y_stride = M * Y_HxW;
    const int filter_stride = filter.numel() / G;

    // The col buffer is stored in CHW order as well - kernel_dim, and the height
    // and width.
    const T* X_data = X.template data<T>();
    const T* filter_data = filter.template data<T>();
    const T* bias_data = nullptr;
    if (InputSize() == 3) {
      const auto& bias = Input(BIAS);
      CAFFE_ENFORCE_EQ(bias.dim(), 1);
      CAFFE_ENFORCE_EQ(bias.dim32(0), M);
      bias_data = bias.template data<T>();
      ConvPoolOpBase<Context>::template SetBiasMultiplier<T>(
          Y_HxW, &bias_multiplier_);
    }
    T* Y_data = Y->template mutable_data<T>();


    const auto func = [&](Tensor* col_buffer) {
      col_buffer->Resize(buffer_shape);
      T* col_buffer_data = col_buffer->template mutable_data<T>();


  std::cout << "verifying inputs to GEMM kernel" << std::endl;

  std::cout << "Weight:" << std::endl;
  std::cout << "kernel_size: "<< kernel_size << std::endl;
  for(int n = 0; n < M; ++n){
    std::cout<<"[";
    for(int c = 0; c < C; ++c){
      std::cout<<"[";
      for(int h = 0; h < 5; ++h){
          std::cout<<"["<<std::endl;
        for(int w = 0; w<5; ++w){
          std::cout<<"[";
          std::cout<<"at n,c,h,w: "<<n<<" "<<c<<" "<<h<<" "<<w<<" ";
          std::cout<<filter_data[w+5*h+25*c+25*C*n]<<"]"<<std::endl;
        }
          std::cout<<"]"<<std::endl;
      }
      std::cout<<"]"<<std::endl;
    }
    std::cout<<"]"<<std::endl;
  }

  std::cout << "Input" << std::endl;
  for(int n = 0; n < N; ++n){
    std::cout<<"[";
    for(int c = 0; c < C; ++c){
      std::cout<<"[";
      for(int h = 0; h < 8; ++h){
          std::cout<<"["<<std::endl;
        for(int w = 0; w < 8; ++w){
          std::cout<<"[";
          std::cout<<"at n,c,h,w: "<<n<<" "<<c<<" "<<h<<" "<<w<<" ";
          std::cout<<X_data[w+8*h+X_HxW*c+X_HxW*C*n]<<"]"<<std::endl;
        }
          std::cout<<"]"<<std::endl;
      }
      std::cout<<"]"<<std::endl;
    }
    std::cout<<"]"<<std::endl;
  }

  std::cout << "Output" << std::endl;


  FPGAGEMM(CblasNoTrans,
            CblasNoTrans,
            M,
            Y_HxW,
            kernel_dim,
            1.0f,
            filter_data,
            X_data,
            0.0f,
            Y_data
            );

  //CHECK: what about col2im

     };
   if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
     runWithSharedBuffer<Context>(ws_, func);
   } else {
     func(&col_buffer_);
   }
   return true;
 }



template <typename T, class Context>
bool Conv_fpga_Op<T, Context>::RunOnDeviceWithOrderNHWC() {
  CAFFE_ENFORCE_LE(
      kernel_.size(),
      3,
      "Only 1-3d convolution is supported for NHWC storage type");
  const Tensor& X = Input(INPUT);
  const auto& filter = Input(FILTER);
  Tensor* Y = Output(0);
  const int N = X.dim32(0), C = X.dim32(X.dim() - 1);
  const int G = group_;
  CAFFE_ENFORCE_EQ(X.dim(), filter.dim());
  const int M = filter.dim32(0);
  CAFFE_ENFORCE_EQ(
      C,
      filter.dim32(filter.dim() - 1) * G,
      "Convolution op: input channels does not match: # of input channels ",
      C,
      " is not equal to kernel channels * group: ",
      filter.dim32(filter.dim() - 1),
      "*",
      G);
  CAFFE_ENFORCE_EQ(
      M % G, 0, "The number of output channels is not divisible by group.");

  int kernel_size = 1;
  for (std::size_t i = 0; i < kernel_.size(); ++i) {
    CAFFE_ENFORCE_EQ(filter.dim32(i + 1), kernel_[i]);
    kernel_size *= kernel_[i];
  }
  ConvPoolOpBase<Context>::SetOutputSize(X, Y, M);

  if (N == 0) {
    Y->template mutable_data<T>();
    return true;
  }

  const vector<int> Y_dims = GetDims(*Y);
  const int X_HxW = X.numel() / (N * C);
  const int Y_HxW = Y->numel() / (N * M);
  const vector<int> img_shape(X.sizes().cbegin() + 1, X.sizes().cend());
  vector<int> buffer_shape(Y_dims.size() + 1);
  std::copy(Y_dims.cbegin(), Y_dims.cend(), buffer_shape.begin());
  buffer_shape.back() = C * kernel_size;

  const int buffer_size = C * kernel_size * Y_HxW;

  // The dimension of each kernel
  const int kernel_dim = C / G * kernel_size;
  // The offset corresponding to a single input image, and a single output
  // image.
  const int input_offset = X_HxW * C;
  const int output_offset = Y->numel() / Y->dim32(0);

  // The output image size is the spatial size of the output.
  // The col buffer is stored in HWC order as well - the height and width, and
  // kernel_dim.
  const T* X_data = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  const T* bias_data = nullptr;
  if (InputSize() == 3) {
    const auto& bias = Input(BIAS);
    CAFFE_ENFORCE_EQ(bias.dim(), 1);
    CAFFE_ENFORCE_EQ(bias.dim32(0), M);
    bias_data = bias.template data<T>();
  }
  T* Y_data = Y->template mutable_data<T>();

  // Specialized path for 1 by 1 convolution with stride 1, pad 0 - we
  // can skip im2col.
  if (kernel_dim == (C / group_) && !HasPad() && !HasStride()) {
    if (bias_data != nullptr) {
      // For this specialized path, we need a bigger bias_multiplier_ because
      // we're doing just 1 big GEMM.
      ConvPoolOpBase<Context>::template SetBiasMultiplier<T>(
          N * X_HxW, &bias_multiplier_);
    }
    return Run1x1ConvOnDeviceWithOrderNHWC(
        N, C, X_HxW, M, X_data, filter_data, bias_data, Y_data);
  }

  if (bias_data != nullptr) {
    ConvPoolOpBase<Context>::template SetBiasMultiplier<T>(
        Y_HxW, &bias_multiplier_);
  }
  auto f = [&](Tensor* col_buffer) {
    col_buffer->Resize(buffer_shape);
    T* col_buffer_data = col_buffer->template mutable_data<T>();
    // Im2Col, followed by gemm.
    for (int image_id = 0; image_id < N; ++image_id) {
      if (kernel_.size() <= 2) {
        math::Im2Col<T, Context, StorageOrder::NHWC>(
            C,
            X.dim32(1),
            kernel_.size() == 2 ? X.dim32(2) : 1,
            kernel_h(),
            kernel_.size() == 2 ? kernel_w() : 1,
            dilation_h(),
            kernel_.size() == 2 ? dilation_w() : 1,
            pad_t(),
            kernel_.size() == 2 ? pad_l() : 0,
            kernel_.size() == 2 ? pad_b() : pad_l(),
            kernel_.size() == 2 ? pad_r() : 0,
            stride_h(),
            kernel_.size() == 2 ? stride_w() : 1,
            X_data,
            col_buffer_data,
            &context_,
            group_);
      } else {
        math::Im2ColNd<T, Context, StorageOrder::NHWC>(
            kernel_.size(),
            C * X_HxW,
            buffer_size,
            img_shape.data(),
            buffer_shape.data(),
            kernel_.data(),
            stride_.data(),
            dilation_.data(),
            pads_.data(),
            X_data,
            col_buffer_data,
            &context_,
            group_);
      }
      // Weight term
      for (int group_id = 0; group_id < group_; ++group_id) {
        // col_buffer_data in G (H W) (R S C/G) layout
        // filter_data in G K/G (R S C/G) layout
        math::GemmEx<T, Context>(
            CblasNoTrans,
            CblasTrans,
            Y_HxW,
            M / group_,
            kernel_dim,
            1,
            col_buffer_data + group_id * kernel_dim,
            group_ * kernel_dim,
            filter_data + group_id * (M / group_) * kernel_dim,
            kernel_dim,
            0,
            Y_data + group_id * (M / group_),
            M,
            &context_);
      }
      if (bias_data != nullptr) {
        // Bias term
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            Y_HxW,
            M,
            1,
            1,
            bias_multiplier_.template data<T>(),
            bias_data,
            1,
            Y_data,
            &context_);
      }
      X_data += input_offset;
      Y_data += output_offset;
    }
  };
  if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
    runWithSharedBuffer<Context>(ws_, f);
  } else {
    f(&col_buffer_);
  }
  return true;
}

template <typename T, class Context>
bool Conv_fpga_Op<T, Context>::Run1x1ConvOnDeviceWithOrderNCHW(
    const int N,
    const int C,
    const int HxW,
    const int M,
    const T* X,
    const T* filter,
    const T* bias,
    T* Y) {
  const int G = group_;
  if (G == 1) {
    math::GemmStridedBatched<T, Context>(
        CblasNoTrans,
        CblasNoTrans,
        N,
        M,
        HxW,
        C,
        1.0f,
        filter,
        0,
        X,
        C * HxW,
        0.0f,
        Y,
        M * HxW,
        &context_);
  } else {
    const int batch_size = N * G;
    const int D_X = C / G;
    const int D_Y = M / G;
    const int X_stride = D_X * HxW;
    const int W_stride = D_Y * D_X;
    const int Y_stride = D_Y * HxW;
    std::vector<const T*> X_ptr(N * G);
    std::vector<const T*> W_ptr(N * G);
    std::vector<T*> Y_ptr(N * G);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < G; ++j) {
        const int index = i * G + j;
        X_ptr[index] = X + index * X_stride;
        W_ptr[index] = filter + j * W_stride;
        Y_ptr[index] = Y + index * Y_stride;
      }
    }
    math::GemmBatched<T, Context>(
        CblasNoTrans,
        CblasNoTrans,
        batch_size,
        D_Y,
        HxW,
        D_X,
        1.0f,
        W_ptr.data(),
        X_ptr.data(),
        0.0f,
        Y_ptr.data(),
        &context_);
  }
  if (bias != nullptr) {
    const T* bias_multiplier_data = bias_multiplier_.template data<T>();
    math::GemmStridedBatched<T, Context>(
        CblasNoTrans,
        CblasNoTrans,
        N,
        M,
        HxW,
        1,
        1.0f,
        bias,
        0,
        bias_multiplier_data,
        0,
        1.0f,
        Y,
        M * HxW,
        &context_);
  }
  return true;
}

template <typename T, class Context>
bool Conv_fpga_Op<T, Context>::Run1x1ConvOnDeviceWithOrderNHWC(
    const int N,
    const int C,
    const int HxW,
    const int M,
    const T* X,
    const T* filter,
    const T* bias,
    T* Y) {
  const int G = group_;
  const int kernel_dim = C / G;
  for (int group_id = 0; group_id < group_; ++group_id) {
    math::GemmEx<T, Context>(
        CblasNoTrans,
        CblasTrans,
        N * HxW,
        M / group_,
        kernel_dim,
        1.0f,
        X + group_id * kernel_dim,
        C,
        filter + group_id * (M / group_) * kernel_dim,
        kernel_dim,
        0.0f,
        Y + group_id * (M / group_),
        M,
        &context_);
  }
  if (bias != nullptr) {
    const T* bias_multiplier_data = bias_multiplier_.template data<T>();
    math::Gemm<T, Context>(
        CblasNoTrans,
        CblasNoTrans,
        N * HxW,
        M,
        1,
        1.0f,
        bias_multiplier_data,
        bias,
        1.0f,
        Y,
        &context_);
  }
  return true;
}

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CONV_OP_FPGA_IMPL_H_
