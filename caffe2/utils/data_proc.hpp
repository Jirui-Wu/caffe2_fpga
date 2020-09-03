#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include "caffe2/utils/mesh_processor.hpp"
#include "caffe2/utils/xcl2.hpp"
#include <vector>
#include <string>
#include <map>
#include <random>
#include <memory>
#include <iostream>

namespace caffe2 {
namespace math {
void load_stats(std::vector<std::string>& layers, std::map< std::string,std::vector<unsigned> >& stats, const std::string& fileName);
void load_ifm(ifm_t* ifmMat, const std::string& fileName, const int batchSize);
void load_weights(w_t* weightsMat, const std::string& fileName);
void load_bias(ofm_t* biasVec);
void exp_out(const w_t* weightsMat, const ifm_t* ifmMat);
int verify_result();
int verify_result(std::vector<ofm_t, aligned_allocator<ofm_t>>& fpgaVec);
void cpu_calc_write_ofm(std::vector<ofm_t, aligned_allocator<ofm_t>> &ofmExp,
        const std::vector<ifm_t, aligned_allocator<ifm_t>> ifmVec,
        const std::vector<ifm_t, aligned_allocator<w_t>> weightsVec,
        const int* params);

void gpu_profiling(std::vector<ifm_t, aligned_allocator<ifm_t>>& ifmMat, const int channels, const int ifmH, const int ifmW,
        const int wK, const int pad, const int stride, const long i2cSize, const int i2cH, const int i2cW,
        const int batchSize, const int ofmSize);

// Template Functions
template <typename matType>
void load_data(std::vector<matType>& inputElements, const std::string fileName);

template <typename matType>
void save_data(const std::vector<matType, aligned_allocator<matType>> vec, const int* params, const std::string fileName);

template <typename matType>
void create_mat(std::string seedVal, matType* mat, const long matSize, const int batchSize, const int value = -1);
}//math
}//caffe2
#endif // DATA_LOADER_HPP
