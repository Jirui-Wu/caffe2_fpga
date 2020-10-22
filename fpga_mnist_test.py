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

# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
print("Necessities imported!")

# If True, use the LeNet CNN model
# If False, a multilayer perceptron model is used
USE_LENET_MODEL = True

# This section preps your image and test set in a lmdb database
# def DownloadResource(url, path):
#     '''Downloads resources from s3 by url and unzips them to the provided path'''
#     import requests, zipfile
#     from io import StringIO
#     print("Downloading... {} to {}".format(url, path))
#     r = requests.get(url, stream=True)
#     z = zipfile.ZipFile(io.StringIO(r.content))
#     z.extractall(path)
#     print("Completed download and extraction.")

# Setup the paths for the necessary directories
current_folder = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks')
data_folder = os.path.join(current_folder, 'tutorial_data', 'mnist')
root_folder = os.path.join(current_folder, 'tutorial_files', 'tutorial_mnist')
db_missing = False

# Check if the data folder already exists
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    print("Your data folder was not found!! This was generated: {}".format(data_folder))

# Check if the training lmdb exists in the data folder
if os.path.exists(os.path.join(data_folder,"mnist-train-nchw-lmdb")):
    print("lmdb train db found!")
else:
    db_missing = True

# Check if the testing lmdb exists in the data folder
if os.path.exists(os.path.join(data_folder,"mnist-test-nchw-lmdb")):
    print("lmdb test db found!")
else:
    db_missing = True

# Attempt the download of the db if either was missing
if db_missing:
    print("one or both of the MNIST lmbd dbs not found!!")
    db_url = "http://download.caffe2.ai/databases/mnist-lmdb.zip"
    try:
        #DownloadResource(db_url, data_folder)
        print("here!!")
        print("Manual. Please download it manually from {}".format(db_url))
        print("Unzip it and place the two database folders here: {}".format(data_folder))
    except Exception as ex:
        print("Failed to download dataset. Please download it manually from {}".format(db_url))
        print("Unzip it and place the two database folders here: {}".format(data_folder))
        raise ex

# Clean up statistics from any old runs
if os.path.exists(root_folder):
    print("Looks like you ran this before, so we need to cleanup those old files...")
    shutil.rmtree(root_folder)

os.makedirs(root_folder)
workspace.ResetWorkspace(root_folder)

print("training data folder:" + data_folder)
print("workspace root folder:" + root_folder)

def AddInput(model, batch_size, db, db_type):
    ### load the data from db - Method 1 using brew
    #data_uint8, label = brew.db_input(
    #    model,
    #    blobs_out=["data_uint8", "label"],
    #    batch_size=batch_size,
    #    db=db,
    #    db_type=db_type,
    #)
    ### load the data from db - Method 2 using TensorProtosDB
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type)
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label

def AddModel(model, data):
    '''
    This part is fpga conv op
    '''
    # Image size: 28 x 28 -> 24 x 24
    conv0 = conv_fpga(model, data, 'conv0', dim_in=1, dim_out=20, kernel=5)
    print("inside add model!!!")
    #conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    # # Image size: 24 x 24 -> 12 x 12
    # pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    # # Image size: 12 x 12 -> 8 x 8
    # conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    # # Image size: 8 x 8 -> 4 x 4
    # pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    # # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    # # Here, the data is flattened from a tensor of dimension 50x4x4 to a vector of length 50*4*4
    # fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
    # relu3 = brew.relu(model, fc3, 'relu3')
    # # Last FC Layer
    # pred = brew.fc(model, relu3, 'pred', dim_in=500, dim_out=10)
    # # Softmax Layer
    # softmax = brew.softmax(model, pred, 'softmax')
    return conv0

#### Train Model
# Specify the data will be input in NCHW order
#  (i.e. [batch_size, num_channels, height, width])
arg_scope = {"order": "NCHW"}
# Create the model helper for the train model
train_model = model_helper.ModelHelper(name="mnist_train", arg_scope=arg_scope)
# Specify the input is from the train lmdb
data, label = AddInput(
    train_model, batch_size=4,
    db=os.path.join(data_folder, 'mnist-train-nchw-lmdb'),
    db_type='lmdb')

# Add the model definition (fc layers, conv layers, softmax, etc.)
conv0 = AddModel(train_model, data)

#### Deployment model.
# We simply need the main AddModel part.
deploy_model = model_helper.ModelHelper(
    name="mnist_deploy", arg_scope=arg_scope, init_params=False)
AddModel(deploy_model, "data")

print("*******train_model.net.Proto()*******\n")
print(str(train_model.net.Proto())[:400] + '\n...')
print("\n*******train_model.param_init_net.Proto()*******\n")
print(str(train_model.param_init_net.Proto())[:400] + '\n...')

# The parameter initialization network only needs to be run once.
# Now all the parameter blobs are initialized in the workspace.
workspace.RunNetOnce(train_model.param_init_net)

# Creating an actual network as a C++ object in memory.
#   We need this as the object is going to be used a lot
#   so we avoid creating an object every single time it is used.
# overwrite=True allows you to run this cell several times and avoid errors
workspace.CreateNet(train_model.net, overwrite=True)

# Set the iterations number and track the accuracy & loss
total_iters = 10

# MAIN TRAINING LOOP!
for i in range(total_iters):
    print("training...")
    workspace.RunNet(train_model.net)



blob = workspace.FetchBlob("data")
print("fetched after 10 runs:\n", blob)
# reset the workspace, to make sure the model is actually loaded
workspace.ResetWorkspace(root_folder)


# feed the previously saved data to the loaded model
print("after reset input:\n", workspace.FetchBlob("data"))
workspace.FeedBlob("data", blob)


print("output:\n", workspace.FetchBlob("conv0"))
print("Shape of conv0: ",conv0.shape)


print("End of testing script")
