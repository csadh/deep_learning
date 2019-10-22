###### import the required package#####
import numpy as np                     #fundamental package for scientific computing with Python
import matplotlib.pyplot as plt        #famous library to plot graphs in Python
import h5py                            #common package to interact with a dataset that is stored on an H5 file
import scipy                           #used here to test your model with your own picture at the end

#############################
#Task to load image dataset##
#############################
def load_dataset():
    with h5py.File('train_catvnoncat.h5', "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    with h5py.File('test_catvnoncat.h5', "r") as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

##############################################
#Loading the data in matrix, reshape & scale##
##############################################
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

plt.imshow(train_set_x_orig[100])
plt.show()
print ("y = " + str(train_set_y[0,index]) + ". It's a " + classes[train_set_y[0,index]].decode("utf-8") +  " picture.")

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten  = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
train_set_x = train_set_x_flatten/255
test_set_x  = test_set_x_flatten/255
##do assertion check on sizes
assert(train_set_x.shape == (train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3],train_set_x_orig.shape[0]) )
assert(test_set_x.shape  == (test_set_x_orig.shape[1] *test_set_x_orig.shape[2] *test_set_x_orig.shape[3] ,test_set_x_orig.shape[0] ) )
assert(train_set_y.shape == (1,train_set_y.shape[1]) )
assert(test_set_y.shape  == (1,test_set_y.shape[1]) )
