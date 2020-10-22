list(APPEND Caffe2_CPU_SRCS
  fpga/AsyncProfiler.cpp
  fpga/mm_fpga.cc
  fpga/xcl2.cpp
  )


# TODO Once all source files are defined inside the local c10_utils_xxx targets,
# it should be the job of the parent CMakeLists.txt to decide what to do with the target (i.e. link it to caffe2)
# instead of us locally adding it to Caffe2_xxx variables.

set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
