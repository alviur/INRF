import numpy as np
import scipy.io as sio


path= '/home/agomez/4_NCNN/3_data/'

train_raw = sio.loadmat(path + 'train_32x32.mat')
test_raw = sio.loadmat(path + 'test_32x32.mat')
extra_raw = sio.loadmat(path + 'extra_32x32.mat')

train = train_raw['X']
trainLabels = train_raw['y']-1

test = test_raw['X']
testLabels = test_raw['y']-1

extra = extra_raw['X']
extraLabels = extra_raw['y']-1

#select 400 samples per class from training
indexTrain = []
samplesClass = np.zeros((10))
index = 0
index2 = 0
trainbatch = np.zeros((32,32,3,4000))
trainbatchLabels = np.zeros((4000))

while (samplesClass.sum()<4000):

    if(samplesClass[trainLabels[index,0]]<400):

        indexTrain.append(index)
        trainbatch[:,:,:,index2] = train[:,:,:,index]
        trainbatchLabels[index2] = trainLabels[index]
        samplesClass[trainLabels[index, 0]] += 1
        index2 += 1

    index += 1

#select 200 samples per class from extra
indexExtra = []
samplesClass = np.zeros((10))
index = 0
index2 = 0
extrabatch = np.zeros((32,32,3,2000))
extrabatchLabels = np.zeros((2000))

while (samplesClass.sum()<2000):

    if(samplesClass[extraLabels[index,0]]<200):

        indexExtra.append(index)
        extrabatch[:,:,:,index2] = extra[:,:,:,index]
        extrabatchLabels[index2] = extraLabels[index]
        samplesClass[extraLabels[index, 0]] += 1
        index2 += 1

    index += 1

# Create validation set
valSet = np.concatenate((trainbatch, extrabatch), axis=3)
valSetLabels = np.concatenate((trainbatchLabels, extrabatchLabels), axis=0)

del trainbatch,extrabatch

# Create training set
trainingSet = np.zeros((32,32,3,598388), dtype=int)
trainingLabels = np.zeros((598388), dtype=int)
index = 0

for i in range(extra.shape[3]):

    if((i not in indexExtra)):

        trainingSet[:,:,:,index] = extra[:,:,:,i]
        trainingLabels[index] = extraLabels[i]
        index += 1

del extra

for i in range(train.shape[3]):

    if ((i not in indexTrain)):
        trainingSet[:, :, :, index] = train[:, :, :, i]
        trainingLabels[index] = trainLabels[i]
        index += 1

trainingSet = np.swapaxes(trainingSet,0,2)
trainingSet = np.swapaxes(trainingSet,1,3)
trainingSet = np.swapaxes(trainingSet,0,1)

valSet = np.swapaxes(valSet,0,2)
valSet = np.swapaxes(valSet,1,3)
valSet = np.swapaxes(valSet,0,1)

test = np.swapaxes(test,0,2)
test = np.swapaxes(test,1,3)
test = np.swapaxes(test,0,1)

np.save(path +'trainSVHN',trainingSet)
np.save(path +'trainSVHN_labeks',trainingLabels)
np.save(path +'valSVHN',valSet)
np.save(path +'valSVHN_labels',valSetLabels)
np.save(path +'testSVHN',test)
np.save(path +'testSVHN_labels',testLabels)