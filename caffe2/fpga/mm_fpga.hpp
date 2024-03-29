#ifndef MM_FPGA_H_
#define MM_FPGA_H_

#include "caffe2/core/common.h"
#include "xcl2.hpp"
#include "math.h"

#include "caffe2/fpga/AsyncProfiler.hpp"
#include <vector>
namespace caffe2 {
namespace math {
CAFFE2_API void Kernel(
  // const CBLAS_TRANSPOSE trans_A,
  // const CBLAS_TRANSPOSE trans_B,
  const int trans_A,
  const int trans_B,
  const int M,
  const int N,
  const int K,
  const float alpha,
  const float* A,
  const float* B,
  const float beta,
  float* C,
  double *fpga_times
);
}//math
}//caffe2
#endif
