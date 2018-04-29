import cv2
import sklearn
from sklearn.cluster import KMeans
import scipy.cluster.vq as vq
import numpy as np



DSIFT_STEP_SIZE = 4

# reference: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
import keras
from keras.datasets import mnist
from keras import backend as K

def load_mnist():
    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def load_cifar10_data(dataset):
    if dataset == 'train':
        with open('./cifar10/train/train.txt','r') as f:
            paths = f.readlines()
    if dataset == 'test':
        with open('./cifar10/test/test.txt','r') as f:
            paths = f.readlines()
    x, y = [], []
    for each in paths:
        each = each.strip()
        path, label = each.split(' ')
        img = cv2.imread(path)
        x.append(img)
        y.append(label)
    return [x, y]

def load_my_data(path, test=None):
    with open(path, 'r') as f:
        paths = f.readlines()
    x, y = [], []
    for each in paths:
        each = each.strip()
        label, path = each.split(' ')
        img = cv2.imread(path)
        if img.shape[:2] != (256,256):
            img = cv2.resize(img, (256,256))
        x.append(img)
        y.append(label)
    return [x, y]

def extract_sift_descriptors(img):
    """
    Input BGR numpy array
    Return SIFT descriptors for an image
    Return None if there's no descriptor detected
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


def extract_DenseSift_descriptors(img):
    """
    Input BGR numpy array
    Return Dense SIFT descriptors for an image
    Return None if there's no descriptor detected
    """
    if img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img[:,:,0];
    sift = cv2.xfeatures2d.SIFT_create()

    # opencv docs DenseFeatureDetector
    # opencv 2.x code
    #dense.setInt('initXyStep',8) # each step is 8 pixel
    #dense.setInt('initImgBound',8)
    #dense.setInt('initFeatureScale',16) # each grid is 16*16
    disft_step_size = DSIFT_STEP_SIZE
    keypoints = [cv2.KeyPoint(x, y, disft_step_size)
            for y in range(0, gray.shape[0], disft_step_size)
                for x in range(0, gray.shape[1], disft_step_size)]

    keypoints, descriptors = sift.compute(gray, keypoints)

    #keypoints, descriptors = sift.detectAndCompute(gray, None)
    return [keypoints, descriptors]


def build_codebook(X, voc_size):
    """
    Inupt a list of feature descriptors
    voc_size is the "K" in K-means, k is also called vocabulary size
    Return the codebook/dictionary
    """
    features = np.vstack((descriptor for descriptor in X))
    kmeans = KMeans(n_clusters=voc_size, n_jobs=-2)
    kmeans.fit(features)
    codebook = kmeans.cluster_centers_.squeeze()
    return codebook


def input_vector_encoder(feature, codebook):
    """
    Input all the local feature of the image
    Pooling (encoding) by codebook and return
    """
    code, _ = vq.vq(feature, codebook)
    word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
    return word_hist
