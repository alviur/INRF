from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np
import scipy.io as sio


def loadMnist():

    mnist = tf.keras.datasets.mnist
    (train_images0, train_labels0), (test_images, test_labels) = mnist.load_data()

    train_images0 = train_images0.astype(np.float32) / 255

    test_images = test_images.astype(np.float32) / 255
    test_images = np.expand_dims(test_images, -1)

    val_images = train_images0[50000:, :]
    val_labels = train_labels0[50000:]
    train_images = train_images0[0:50000]
    train_labels = train_labels0[0:50000]

    train_images = np.expand_dims(train_images, -1)
    train_images0 = np.expand_dims(train_images0, -1)
    val_images = np.expand_dims(val_images, -1)

    return train_labels,train_labels0,test_labels,val_labels,train_images,val_images,test_images,train_images0

def loadMnist2():

    mnist = tf.keras.datasets.mnist
    (train_images0, train_labels0), (test_images, test_labels) = mnist.load_data()

    train_images0 = train_images0.astype(np.float32) / 255

    test_images = test_images.astype(np.float32) / 255
    test_images = np.expand_dims(test_images, -1)

    val_images = train_images0[55000:, :]
    val_labels = train_labels0[55000:]
    train_images = train_images0[0:55000]
    train_labels = train_labels0[0:55000]

    train_images = np.expand_dims(train_images, -1)
    train_images0 = np.expand_dims(train_images0, -1)
    val_images = np.expand_dims(val_images, -1)

    # One hot  encoding
    depth = 10
    enc = OneHotEncoder(n_values=depth, sparse=False)
    #enc.fit(test_labels.reshape(-1, 1))
    #train_labels = enc.transform(train_labels.reshape(-1, 1))
    #train_labels0 = enc.transform(train_labels0.reshape(-1, 1))
    #test_labels = enc.transform(test_labels.reshape(-1, 1))
    #val_labels = enc.transform(val_labels.reshape(-1, 1))

    return train_labels,train_labels0,test_labels,val_labels,train_images,val_images,test_images,train_images0

def loadMnist3(maxTrain = 50000):

    mnist = tf.keras.datasets.mnist
    (train_images0, train_labels0), (test_images, test_labels) = mnist.load_data()

    train_images0 = train_images0.astype(np.float32) / 255

    test_images = test_images.astype(np.float32) / 255
    test_images = np.expand_dims(test_images, -1)

    val_images = train_images0[55000:, :]
    val_labels = train_labels0[55000:]
    train_images = train_images0[0:maxTrain]
    train_labels = train_labels0[0:maxTrain]

    train_images = np.expand_dims(train_images, -1)
    train_images0 = np.expand_dims(train_images0, -1)
    val_images = np.expand_dims(val_images, -1)

    # One hot  encoding
    depth = 10
    enc = OneHotEncoder(n_values=depth, sparse=False)
    #enc.fit(test_labels.reshape(-1, 1))
    #train_labels = enc.transform(train_labels.reshape(-1, 1))
    #train_labels0 = enc.transform(train_labels0.reshape(-1, 1))
    #test_labels = enc.transform(test_labels.reshape(-1, 1))
    #val_labels = enc.transform(val_labels.reshape(-1, 1))

    return train_labels,train_labels0,test_labels,val_labels,train_images,val_images,test_images,train_images0

def loadSVHN(path):

    train_raw = sio.loadmat(path + 'trainGray.mat')
    trainLabels_r = sio.loadmat(path + 'train_32x32.mat')
    testLabels_r = sio.loadmat(path + 'test_32x32.mat')
    test_raw = sio.loadmat(path + 'testGray.mat')

    train_X = train_raw['Xgray']
    train_y = trainLabels_r['y'] - 1
    test_X = test_raw['Xgray']
    test_y = testLabels_r['y'] - 1

    train_images0 = train_X.astype(np.float32) / 255

    test_images = test_X.astype(np.float32) / 255
    test_images = np.expand_dims(test_images, -1)

    val_images = train_images0[60000:, :]
    val_labels = train_y[60000:]
    train_images = train_images0[0:60000]
    train_labels = train_y[0:60000]

    train_images = np.expand_dims(train_images, -1)
    train_images0 = np.expand_dims(train_images0, -1)
    val_images = np.expand_dims(val_images, -1)

    # One hot  encoding
    depth = 10
    enc = OneHotEncoder(n_values=depth, sparse=False)
    enc.fit(test_y.reshape(-1, 1))
    train_labels = enc.transform(train_labels.reshape(-1, 1))
    train_labels0 = enc.transform(train_labels.reshape(-1, 1))
    test_labels = enc.transform(test_y.reshape(-1, 1))
    val_labels = enc.transform(val_labels.reshape(-1, 1))

    return train_labels,train_labels0,test_labels,val_labels,train_images,val_images,test_images,train_images0
