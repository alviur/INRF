#############################################################
# Training code for SVHN dataset
#############################################################


# Libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import INRF2_loop as INRF
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.misc

# Hardware
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
path = '/home/agomez/4_NCNN/3_data/'# Path to data processed with dataLoader_SVHN.py

train_images = np.load(path +'trainSVHN.npy')/255
train_labels = np.load(path +'trainSVHN_labels.npy')
val_images = np.load(path +'valSVHN.npy')/255
val_labels = np.load(path +'valSVHN_labels.npy')
test = np.load(path +'testSVHN.npy')/255
testLabels = np.squeeze(np.load(path +'testSVHN_labels.npy'))

# Path to save model
fileName = '/home/agomez/4_NCNN/1_algorithms/SVHN_INRF.pth'

# Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inrfLayer1 = INRF.INRF(dimG=1, dimW=5, numChanIn=3, numChanOut=192, sigma=1, paramSigma=100, lambdaV = 1.0)
        self.inrfLayer2 = INRF.INRF(dimG=1, dimW=5, numChanIn=192, numChanOut=192, sigma=1, paramSigma=100, lambdaV= 1.0)
        self.inrfLayer3 = INRF.INRF(dimG=1, dimW=5, numChanIn=192, numChanOut=192, sigma=1, paramSigma=100, lambdaV=1.0)

        # ConvNet # Uncomment for CNN version
        '''self.conv1 = nn.Conv2d(in_channels=3, out_channels=192, kernel_size=5, padding=(5 // 2, 5 // 2), bias=True)
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=(5 // 2, 5 // 2), bias=True)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=(5 // 2, 5 // 2), bias=True)'''

        self.pool = nn.MaxPool2d(2, 2)
        self.fc3 = nn.Linear(192, 10)

        self.bn1 = nn.BatchNorm2d(192)
        self.bn2 = nn.BatchNorm2d(192)
        self.bn3 = nn.BatchNorm2d(192)

        self.avgpool = nn.AvgPool2d(8,8)

    def forward(self, x):

        #print(1. / math.sqrt(self.conv1.weight.size(1)))

        # I3N
        x00 = self.pool(self.bn1(F.relu(self.inrfLayer1.forward((x)))))
        x01 = self.bn2(F.relu(self.inrfLayer2.forward((x00))))
        x03 = self.pool(x01)
        x04 = self.bn3(F.relu(self.inrfLayer3.forward((x03))))
        x05 = self.avgpool((x04))

        # ConvNet # Uncomment for CNN version
        '''x00 = self.pool((self.bn1(F.relu(self.conv1((x))))))
        x01 = self.bn2(F.relu(self.conv2((x00))))
        x03 = self.pool(x01)
        x04 = self.bn3(self.conv3((x03)))
        x05 = self.avgpool((x04))'''


        x = x05.view(-1, 192)
        x = self.fc3(x)

        return x


net = Net()
net = net.to(device)
#net.cuda()


# Optimizer
criterion = nn.CrossEntropyLoss()
wd = 0.00003 # weight decay
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=wd, amsgrad = True)

batch_size = 64
trainingImag =598388
training_iters = (trainingImag)*100 # Equivalent to 200 epochs
display_step = 200
step = 1
bestTest = 100
batchCont = 0
patienceCont1 = 0
running_loss = 0
test_step = 200000
batch = 0
while step * batch_size < training_iters:
    net.train()
    if (batchCont + batch_size < trainingImag):

        batch_x = train_images[batchCont:batchCont + batch_size, :]
        batch_y = train_labels[batchCont:batchCont + batch_size].astype(int)

        batchCont = batchCont + batch_size

    else:

        batch_x = train_images[batchCont:trainingImag, :]
        batch_y = train_labels[batchCont:trainingImag].astype(int)

        batchCont = 0

    inputs0 = (batch_x)
    inputs0 = (inputs0)

    inputs = torch.Tensor(inputs0).cuda()
    labels = torch.Tensor(batch_y).cuda()

    # Run optimization op (backprop)
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)

    #print(labels.long())
    loss = criterion(outputs, labels.long())


    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()


    if step % display_step == 0:
        print("Epoch:",int((step * batch_size)/trainingImag)," Loss:",running_loss / display_step)
        running_loss = 0.0

    step += 1

    if step * batch_size > test_step:

        test_step += 200000

        # Validation error
        batchTest = 0
        testSize = 6000
        correct = 0
        total = 0
        while batchTest < testSize:
            net.eval()
            # batch_x, batch_y = mnist.train.next_batch(batch_size)

            if (batchTest + batch_size < testSize):

                batch_x = val_images[batchTest:batchTest + batch_size, :]
                batch_y = val_labels[batchTest:batchTest + batch_size]

                batchTest = batchTest + batch_size

            else:

                batch_x = val_images[batchTest:, :]
                batch_y = val_labels[batchTest:]
                batchTest = batchTest + batch_size

            inputs0 = (batch_x)
            inputs0 = (inputs0)

            inputs = torch.Tensor(inputs0).cuda()
            labels = torch.Tensor(batch_y).cuda()

            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()

        accCum = correct / total

        if (bestTest < 100 - 100 * accCum):

            print("Best validation Error:", bestTest)
            patienceCont1 += 1
        else:
            print("validation Error:", 100 - 100 * accCum)
            bestTest = 100 - 100 * accCum
            # Save model
            PATH = fileName
            torch.save(net.state_dict(), PATH)

            patienceCont = 0

    if (int((step * batch_size)/trainingImag)>batch):

        batch += 1

        # Test error
        net.eval()
        batchTest = 0
        testSize = 26032
        accCum = 0
        accCont = 0
        correct = 0
        total = 0
        while batchTest < testSize:
            net.eval()
            # batch_x, batch_y = mnist.train.next_batch(batch_size)

            if (batchTest + batch_size < testSize):

                batch_x = test[batchTest:batchTest + batch_size, :]
                batch_y = testLabels[batchTest:batchTest + batch_size]

                batchTest = batchTest + batch_size

            else:

                batch_x = test[batchTest:, :]
                batch_y = testLabels[batchTest:]
                batchTest = batchTest + batch_size

            inputs0 = (batch_x)
            inputs0 = (inputs0)

            inputs = torch.Tensor(inputs0).cuda()
            labels = torch.Tensor(batch_y).cuda()

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()

        accCum = correct / total
        print("==================================")
        print("Testing error:", 100 - 100 * accCum)
        print("==================================")

# Test error best validation accuracy
PATH = fileName
net = Net()
net.load_state_dict(torch.load(PATH))
net = net.to(device)
net.eval()
batchTest = 0
testSize = 26032
accCum = 0
accCont = 0
correct = 0
total = 0
while batchTest < testSize:
    net.eval()
    # batch_x, batch_y = mnist.train.next_batch(batch_size)

    if (batchTest + batch_size < testSize):

        batch_x = test[batchTest:batchTest + batch_size, :]
        batch_y = testLabels[batchTest:batchTest + batch_size]

        batchTest = batchTest + batch_size

    else:

        batch_x = test[batchTest:, :]
        batch_y = testLabels[batchTest:]
        batchTest = batchTest + batch_size

    inputs0 = (batch_x)
    inputs0 = (inputs0)

    inputs = torch.Tensor(inputs0).cuda()
    labels = torch.Tensor(batch_y).cuda()

    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.long()).sum().item()
    #print(inputs0.shape,predicted.shape,labels.long().shape,(predicted == labels.long()).sum().item())

print(correct,total)
accCum = correct / total
print("==================================")
print("Testing error:", 100 - 100 * accCum)
print("==================================")

