from utils import load_my_data
from utils import extract_DenseSift_descriptors
from utils import build_codebook
from utils import input_vector_encoder
from classifier import svm_classifier
import spm

import numpy as np
import argparse

from utils import load_mnist

parser = argparse.ArgumentParser(description='Image loader')
parser.add_argument("--train", type=str, help='load training data')
parser.add_argument("--test", type=str, help='load test data (optional)')

args = parser.parse_args()
print ("Training data load from {}".format(args.train))
print ("Test data load from {}".format(args.test))

#X, y = load_my_data(args.train)
# to save time...
#X = [X[i] for i in range(0,1000,100)]
#y = [y[i] for i in range(0,1000,100)]
#X = np.array(X[:200])
#y = np.array(y[:200])

(x_train, y_train), (x_test, y_test) = load_mnist()
x_train = np.uint8(x_train*255)
x_test = np.uint8(x_test*255)

x_train_1 = np.reshape(x_train,(-1,28*28))
y_train_1 = np.where(y_train==1)[1]
x_test_1 = np.reshape(x_test,(-1,28*28))
y_test_1 = np.where(y_test==1)[1]

print ("Codebook Size: {:d}".format(spm.VOC_SIZE))
print ("Pyramid level: {:d}".format(spm.PYRAMID_LEVEL))
print ("Dense SIFT feature extraction")

X = x_train
x_feature = [extract_DenseSift_descriptors(img) for img in X]
x_kp, x_des = zip(*x_feature)

print ("Building the codebook, it will take some time")
codebook = build_codebook(x_des, spm.VOC_SIZE)
#import cPickle
#with open('./data/codebook_spm.pkl','w') as f:
#    cPickle.dump(codebook, f)

print ("Spatial Pyramid Matching encoding")
X = [spm.spatial_pyramid_matching(X[i],
                              x_des[i],
                              codebook,
                              level=spm.PYRAMID_LEVEL)
                              for i in range(len(x_des))]
X_train = np.asarray(X)

X = x_test
x_feature = [extract_DenseSift_descriptors(img) for img in X]
x_kp, x_des = zip(*x_feature)
X = [spm.spatial_pyramid_matching(X[i],
                              x_des[i],
                              codebook,
                              level=spm.PYRAMID_LEVEL)
                              for i in range(len(x_des))]
X_test = np.asarray(X)

svm_classifier(X_train, y_train_1, X_test, y_test_1)
