// conv_op_fpga_impl.h is the templated implementation of the conv_op.h file.
#ifndef CAFFE2_OPERATORS_CONV_OP_FPGA_IMPL_H_
#define CAFFE2_OPERATORS_CONV_OP_FPGA_IMPL_H_
//FROM conv_op_fpga_impl
#include "caffe2/operators/conv_fpga_op.h"

#include <array>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

//FROM host.hpp
//#define PY_SSIZE_T_CLEAN
#include "caffe2/utils/mesh_processor.hpp"
#include "caffe2/utils/data_proc.hpp"
#include "caffe2/utils/utils.hpp"
#include "caffe2/utils/xcl2.hpp"
//#include <vector>
//#define DATA_SIZE 4096
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
#include "caffe2/utils/mm_fpga.hpp"

namespace caffe2 {

union cast_t {
    public:
        float f;
        unsigned i;
};

void FPGAGEMM (
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C
  );

template <typename T, class Context>
bool Conv_fpga_Op<T, Context>::RunOnDeviceWithOrderNCHW(){
    std::string modelName = "lenet";
	  std::cout << "Staring here" << std::endl;
    std::cout << "Loading stats ..." << std::endl;
    std::string fileName = "data/" + modelName + "_imagenet/";
    std::map< std::string, std::vector<unsigned> > stats;
    std::vector<std::string> layers;
    math::load_stats(layers, stats, fileName);
    std::cout << "Loading input feature maps and weights sizes and stats ..." << std::endl;
    auto layerNum = 0;
    std::string layerName = layers[layerNum];
    int batchSize = 128;
    long outputSize;
    long inputSize;
    long weightSize;
    long ifmSize;
    long ofmSize;
    long i2cSize;
    weightSize = 1;
    ifmSize = 1;
    ofmSize = 1;
    i2cSize = 1;
    for (const auto& elem: stats[layerName + "-kernel"]) {weightSize *= elem;}
    for (const auto& elem: stats[layerName + "-input"]) {ifmSize *= elem;}
    for (const auto& elem: stats[layerName + "-im2col"]) {i2cSize *= elem;}
    for (const auto& elem: stats[layerName + "-output"]) {ofmSize *= elem;}
    inputSize = ifmSize * long(batchSize);
    outputSize = ofmSize * long(batchSize);
    unsigned channels = stats[layerName + "-input"][0];
    unsigned ifmH = stats[layerName + "-input"][1];
    unsigned ifmW = stats[layerName + "-input"][2];
    unsigned D = stats[layerName + "-kernel"][0];
    unsigned wK = stats[layerName + "-kernel"][1];
    assert(wK == stats[layerName + "-kernel"][2]);
    assert(channels == stats[layerName + "-kernel"][3]);
    unsigned ofmH = stats[layerName + "-output"][0];
    unsigned ofmW = stats[layerName + "-output"][1];
    unsigned ofmC = stats[layerName + "-output"][2];
    assert(ofmC == D);
    unsigned i2cW = stats[layerName + "-i2c"][0];
    unsigned i2cH = stats[layerName + "-i2c"][1] * stats[layerName + "-i2c"][2];
    unsigned stride = stats[layerName + "-stride"][0];
    unsigned pad = stats[layerName + "-pad"][0];
    // //org = 0 is for test purpose
    // auto org = 0;
    // if (org == 0) {
    // batchSize = 12;
    // ifmSize = 8*8*3;
    // inputSize = 8*8*3*12;
    // weightSize = 4*4*6*3;
    // ofmSize = 4*4*6;
    // outputSize= 4*4*6*12;
    // i2cSize = 16*48;
    // channels = 3;
    // ifmH = 8;
    // ifmW = 8;
    // D = 6;
    // wK = 4;
    // ofmH = 4;
    // ofmW = 4;
    // ofmC = 6;
    // i2cW = 16;
    // i2cH = 48;
    // stride = 2;
    // pad = 1;
    // }
    std::cout << "is: " << inputSize << " os: " << outputSize << " ws: " << weightSize << std::endl;

  std::cout << "Loading input feature maps and weights in matrices A and B ..." << std::endl;

  //Question: generating random input fm and weights, this is more like for testing??
  std::vector<ofm_t, aligned_allocator<ofm_t>> ofmMat(outputSize);
  std::vector<ifm_t, aligned_allocator<ifm_t>> ifmMat(inputSize);
  std::vector<w_t, aligned_allocator<w_t>> weightsMat(weightSize);

  std::string seed("5");
  math::create_mat(seed, weightsMat.data(), weightSize, 1);
  assert(weightsMat.size() == weightSize);
  std::cout << "Weight" << std::endl;

  math::create_mat(seed, ifmMat.data(), ifmSize, batchSize, -1);
  assert(ifmMat.size() == ifmSize*batchSize);
  std::cout << "Input" << std::endl;

  math::create_mat(seed, ofmMat.data(), ofmSize, batchSize, 0);
  assert(ofmMat.size() == ofmSize*long(batchSize));
  std::cout << "Output" << std::endl;

  //arguments fro math::Gemm
  int M = D;
  int Y_HxW = i2cW;
  int kernel_dim = i2cH;
  //note here inputs filter_data and col_buffer_data needs to be arrays
  //output Y_data is array by design in math_fpga.cc
  w_t* filter_data;
  ifm_t* col_buffer_data;
  for (int i=0; i< weightsMat.size(); ++i)
  {
      filter_data[i] = weightsMat[i];
  }
  for (int i=0; i< ifmMat.size(); ++i)
  {
      col_buffer_data[i] = ifmMat[i];
  }
  float* Y_data;
  FPGAGEMM(CblasNoTrans,
            CblasNoTrans,
            M,
            Y_HxW,
            kernel_dim,
            1.0f,
            filter_data,
            col_buffer_data,
            0.0f,
            Y_data
            );

  //CHECK: what about col2im
  return true;
}

//added by jirui
void FPGAGEMM(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C
    ){
      //profiling
     //  #ifdef PROFILING
     //  std::ofstream profilingLog;
     //  //TODO: change a path
     //  profilingLog.open("~/home/log.csv", ios::app);
     //  profilingLog << M << "," << N << "," << K << ",";
     //
     //  //from line 276 and 277 of math_cpu.cc, format slightly different, now min is 1
     //  // int lda = (trans_A == CblasNoTrans) ? K : M;
     //  // int ldb = (trans_B == CblasNoTrans) ? N : K;
     // const int lda = std::max((trans_A == CblasNoTrans) ? K : M, 1);
     // const int ldb = std::max((trans_B == CblasNoTrans) ? N : K, 1);
     //
     // caffe2::Timer cpu;
     // double cpu_time = 0.0;
     // cpu.Start();
     //
     // cblas_sgemm(CblasRowMajor, trans_A, trans_B, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
     // cpu_time += cpu.MicroSeconds();
     // profilingLog << cpu_time / 1000.0 << ",";
     // //record results from cblas_sgemm in cBlas
     // float cBlas[N*M];
     // for (int i=0; i<M; i++)
     // {
     //     for (int j=0; j<N; j++)
     //     {
     //         cBlas[i*N + j] = C[i*N+j];
     //     }
     // }
     // #endif

     double fpga_times[3];

     //CHECK: Is this better to be put before the kernel in mm_fpga.cc?
     // std::cout << "Setting up interfaces..." << std::endl;
     // std::vector<cl::Device> devices = xcl::get_xil_devices();
     // cl::Device device = devices[0];
     //
     // cl_int err;
     // OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
     // OCL_CHECK(err, std::string deviceName = device.getInfo<CL_DEVICE_NAME>(&err));
     // //TODO:binary file
     // unsigned fileBufSize;
     // char* fileBuf = xcl::read_binary_file("matmul.xclbin", fileBufSize);
     // cl::Program::Binaries bins{{fileBuf,fileBufSize}};
     // devices.resize(1);
     // OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
     // OCL_CHECK(err, cl::Kernel kernel(program, "matmul", &err));

     //passing by reference will not improve efficiency as A,B and C are arrays in form of ptrs
     //trans_A, trans_B, M,K,N, alpha and beta are constants
     //Kernel(trans_A - 111, A, trans_B - 111, B, C, M, K, K, N, alpha, beta, fpga_times);
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
     // #ifdef PROFILING
     // #ifdef PROFILING_TIME
     // double total_time = 0.0;
     // for (unsigned i=0; i<3; ++i)
     // {
     //     total_time += fpga_times[i];
     // }
     // profilingLog << fpga_times[0] << "," << fpga_times[1] << "," << fpga_times[2] << "," << total_time << ",";
     // #endif
     // //mean square error
     // double mse = 0;
     // for (int i=0; i<M; i++)
     // {
     //     for (int j=0; j<N; j++)
     //     {
     //       mse += std::pow(std::fabs(cBlas[i*N+j] - C[i*N+j]) ,2);
     //     }
     // }
     // mse /= (N*M);
     //
     // profilingLog << mse << std::endl;
     // profilingLog.close();
     // #endif
   }


// The implementations.

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
