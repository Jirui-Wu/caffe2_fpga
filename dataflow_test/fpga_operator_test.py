from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
#%matplotlib inline
#from matplotlib import pyplot
import numpy as np
import os
import shutil
import operator
import caffe2.python.predictor.predictor_exporter as pe

from caffe2.python import (
    brew,
    core,
    model_helper,
    optimizer,
    workspace,
)
# used in caffe2_tutorials directory
# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
print("Necessities imported!")

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Conv_fpga",
    ["X", "filter", "bias"],
    ["Y"],
    kernel=5,
    pad=1,
    stride=2
)

#Create X: (N,C,H,W)
data = np.random.randn(1,3,8,8).astype(np.float32)
print("Data shape: ",data.shape)

#Create W: (M,C,Kh,Kw)
filters = np.random.randn(2,3,5,5).astype(np.float32)
print("Filter shape: ",filters.shape)

#Create b: M
bias = np.array([1.,1.]).astype(np.float32)
print("Bias shape: ",bias.shape)

#Put the inputs into the workspace
workspace.FeedBlob("X", data)
print("X:\n", workspace.FetchBlob("X"))
workspace.FeedBlob("filter", filters)
print("filter:\n", workspace.FetchBlob("filter"))
workspace.FeedBlob("bias", bias)
print("bias:\n", workspace.FetchBlob("bias"))

print("Input genrated, now running the operator.")
#Run the operator
workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))

print("End of testing script")
