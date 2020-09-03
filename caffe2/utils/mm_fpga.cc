#include "caffe2/utils/mm_fpga.hpp"
#include <fstream>
#include <iostream>


namespace caffe2 {

namespace math {

C10_EXPORT void Kernel(
  const CBLAS_TRANSPOSE trans_A,
  const CBLAS_TRANSPOSE trans_B,
  const int M,
  const int N,
  const int K,
  const float alpha,
  const float* A,
  const float* B,
  const float beta,
  float* C,
  double *fpga_times
)
{
  // //CHECK: This is now in core/context_fpga.h
  std::cout << "Setting up interfaces..." << std::endl;
  std::vector<cl::Device> g_devices = xcl::get_xil_devices();
  cl::Device device = g_devices[0];
  cl_int err;
  cl::Context context(device, NULL, NULL, NULL, &err);
  std::string deviceName = device.getInfo<CL_DEVICE_NAME>(&err);

  unsigned fileBufSize;
  //TODO: need to be update to a new binary file
  char* fileBuf = xcl::read_binary_file("matmul.xclbin", fileBufSize);
  cl::Program::Binaries bins{{fileBuf,fileBufSize}};

  cl::Program program(context, std::vector<cl::Device>{device}, bins, NULL, &err);
  cl::Kernel krnl_mesh_proc(program, "matmul", &err);

  OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));


  //A and B here are arrays, change them to vectors, CHECK HERE
  std::vector<float, aligned_allocator<float> > weightsMat{A, A + M*K};
  std::vector<float, aligned_allocator<float> > ifmMat{B, B + N*K};
  std::vector<float, aligned_allocator<float> > ofmMat;
    for (int i=0; i< M*N; ++i)
    {
        ofmMat.push_back(0);
    }

  std::cout << "Creating buffers..." << std::endl;
  OCL_CHECK(err, cl::Buffer weightsMatLoco(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*weightsMat.size(), weightsMat.data(), &err));
  OCL_CHECK(err, cl::Buffer ifmMatLoco(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*ifmMat.size(), ifmMat.data(), &err));
  OCL_CHECK(err, cl::Buffer ofmMatLoco(context, CL_MEM_USE_HOST_PTR, sizeof(float)*ofmMat.size(), ofmMat.data(), &err));

  q.finish();

  //Question: where  to set up the interface
  std::cout << "Setting kernel arguments..." << std::endl;
  OCL_CHECK(err, err = krnl_mesh_proc.setArg(0, weightsMatLoco));
  OCL_CHECK(err, err = krnl_mesh_proc.setArg(1, ifmMatLoco));
  OCL_CHECK(err, err = krnl_mesh_proc.setArg(2, ofmMatLoco));

  std::cout << "Performing Matmul on FPGA..." << std::endl;
  cl::Event wWrite, ifmWrite, ofmWrite, kernelExec, wRead, ifmRead, ofmRead;
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({weightsMatLoco}, 0, NULL, &wWrite));
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({ifmMatLoco}, 0, NULL, &ifmWrite));
  //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({ofmMatLoco}, 0, NULL, &ofmWrite));

  std::vector<cl::Event> kernel_wait_events = {wWrite, ifmWrite};//, ofmWrite};
  OCL_CHECK(err, err = q.enqueueTask(krnl_mesh_proc, &kernel_wait_events, &kernelExec));

  q.finish();

  std::vector<cl::Event> read_wait_events = {kernelExec};
  //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({weightsMatLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &wRead));
  //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({ifmMatLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &ifmRead));
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({ofmMatLoco}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait_events, &ofmRead));

  q.finish();

  std::cout << "Kernel Finished ..." << std::endl;

  #ifdef PROFILING_TIME
  int teSize = 7;
  cl::Event transfer_events[teSize];
  transfer_events[0] = wWrite;
  transfer_events[1] = ifmWrite;
  transfer_events[2] = ofmWrite;
  transfer_events[3] = kernelExec;
  transfer_events[4] = wRead;
  transfer_events[5] = ifmRead;
  transfer_events[6] = ofmRead;

  cl_ulong time_start, time_end;
  std::vector<cl_ulong> event_times;
  for (unsigned i=0; i<teSize; i++)
  {
    OCL_CHECK(err, err = transfer_events[i].wait());
    OCL_CHECK(err, err = transfer_events[i].getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &time_start));
    OCL_CHECK(err, err = transfer_events[i].getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &time_end));
    event_times.push_back(time_end - time_start);
  }

  fpga_times[0] = (event_times[0] + event_times[1] + event_times[2]) / 1000000.0;
  fpga_times[1] = (event_times[3]) / 1000000.0;
  fpga_times[2] = (event_times[4] + event_times[5] + event_times[6] / 1000000.0;
#endif
  //CHANGE vectors back to arrays
  for (int i=0; i< ofmMatLoco.size(); ++i)
  {
      C[i] = ofmMatLoco[i];
  }
  //end of GEMM Kernel
}
}//math
}//caffe2
