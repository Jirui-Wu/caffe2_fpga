if(USE_OPENCL)
  # set(Caffe2_CONTRIB_OPENCL_CPU_SRC
    # "${CMAKE_CURRENT_SOURCE_DIR}/context.cc"
  # )
  # set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${Caffe2_CONTRIB_OPENCL_CPU_SRC} PARENT_SCOPE)
endif()
