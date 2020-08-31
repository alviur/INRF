#!/usr/bin/env python
# Libraries

import numpy as np
import cv2
import matplotlib.pyplot as plt
import INRF2_loop_staticpq as INRF
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import load_data
import math


def load_cnnMNIST(device, filepath):
    # Architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.inrfLayer1 = INRF.INRF(dimG=1, dimW=5, numChanIn=1, numChanOut=32, sigma=1, paramSigma=0.01, lambdaV = 1.1)
            self.inrfLayer2 = INRF.INRF(dimG=1, dimW=5, numChanIn=32, numChanOut=64, sigma=1, paramSigma=0.01, lambdaV= 1.1)

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(5 // 2, 5 // 2), bias=True)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(5 // 2, 5 // 2), bias=True)

            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 500)
            self.dp = nn.Dropout(p=0.5)
            self.dp2d = nn.Dropout2d(p=0.5)
            self.fc2 = nn.Linear(500, 10)

            #self.bn1 = nn.BatchNorm2d(3)
            #self.bn1 = nn.BatchNorm2d(3)

        def forward(self, x):

            # CNN
            x00 = self.pool(F.relu(self.conv1((x))))
            x01 = self.pool(F.relu(self.conv2((x00))))
            x = x01.view(-1, 64 * 7 * 7)



            x = F.relu(self.fc1(x))
            x = self.dp(x)
            x = self.fc2(x)

            return x

    net = Net()
    net.load_state_dict(torch.load(filepath))
    net = net.to(device)
    net.eval()

    return net


def load_arc044_ReLu(device, filepath):

    # Architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.inrfLayer1 = INRF.INRF(dimG=1, dimW=5, numChanIn=1, numChanOut=32, sigma=1, paramSigma=0.01,
                                        lambdaV=1.1)
            self.inrfLayer2 = INRF.INRF(dimG=1, dimW=5, numChanIn=32, numChanOut=64, sigma=1, paramSigma=0.01,
                                        lambdaV=1.1)

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(5 // 2, 5 // 2), bias=False)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(5 // 2, 5 // 2), bias=False)

            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 500)
            self.dp = nn.Dropout(p=0.5)
            self.dp2d = nn.Dropout2d(p=0.5)
            self.fc2 = nn.Linear(500, 10)

            #self.bn1 = nn.BatchNorm2d(3)
            #self.bn1 = nn.BatchNorm2d(3)

        def forward(self, x):

            #print(1. / math.sqrt(self.conv1.weight.size(1)))

            # I3N
            x00 =(self.inrfLayer1.forward((x)))
            x01 = self.pool((x00))
            x02 = (self.inrfLayer2.forward((x01)))
            x2 = self.pool(self.dp2d(x02))
            # CNN

            '''x00 = self.pool(F.relu(self.conv1((x))))
            x01 = self.pool(F.relu(self.conv2((x00))))'''

            x = x2.view(-1, 64 * 7 * 7)

            x = F.relu(self.fc1(x))
            x = self.dp(x)
            x = self.fc2(x)

            return x

    net = Net()
    net = net.to(device)
    net.load_state_dict(torch.load(filepath))
    net = net.to(device)
    net.eval()

    return net

def load_arc046_PQ(device, filepath):
    # Architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.inrfLayer1 = INRF.INRF(dimG=1, dimW=5, numChanIn=1, numChanOut=32, sigma=3, paramSigma=0.01, lambdaV = 1.1)
            self.inrfLayer2 = INRF.INRF(dimG=1, dimW=5, numChanIn=32, numChanOut=64, sigma=1, paramSigma=0.01, lambdaV= 1.1)
            #1. / math.sqrt(1)

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(5 // 2, 5 // 2), bias=False)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(5 // 2, 5 // 2), bias=False)

            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 500)
            self.dp = nn.Dropout(p=0.5)
            self.dp2d = nn.Dropout2d(p=0.5)
            self.fc2 = nn.Linear(500, 10)

            #self.bn1 = nn.BatchNorm2d(3)
            #self.bn1 = nn.BatchNorm2d(3)

        def forward(self, x):

            #print(1. / math.sqrt(self.conv1.weight.size(1)))

            # I3N
            x00 =(self.inrfLayer1.forward((x)))
            x01 = self.pool((x00))
            x02 = (self.inrfLayer2.forward((x01)))
            x2 = self.pool(self.dp2d(x02))
            # CNN

            '''x00 = self.pool(F.relu(self.conv1((x))))
            x01 = self.pool(F.relu(self.conv2((x00))))'''

            x = x2.view(-1, 64 * 7 * 7)

            x = F.relu(self.fc1(x))
            x = self.dp(x)
            x = self.fc2(x)

            return x

    net = Net()
    net = net.to(device)

    net.load_state_dict(torch.load(filepath))
    net = net.to(device)
    net.eval()

    return net


def load_arc_043_ReLu(device, filepath):
       # Architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.inrfLayer1 = INRF.INRF(dimG=1, dimW=5, numChanIn=1, numChanOut=32, sigma=1, paramSigma=100, lambdaV = 1.1)
            self.inrfLayer2 = INRF.INRF(dimG=1, dimW=5, numChanIn=32, numChanOut=64, sigma=1, paramSigma=100, lambdaV= 1.1)
            #1. / math.sqrt(1)

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(5 // 2, 5 // 2), bias=False)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(5 // 2, 5 // 2), bias=False)

            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 500)
            self.dp = nn.Dropout(p=0.5)
            self.dp2d = nn.Dropout2d(p=0.5)
            self.fc2 = nn.Linear(500, 10)

            #self.bn1 = nn.BatchNorm2d(3)
            #self.bn1 = nn.BatchNorm2d(3)

        def forward(self, x):

            #print(1. / math.sqrt(self.conv1.weight.size(1)))

            # I3N
            x00 =(self.inrfLayer1.forward((x)))
            x01 = self.pool((x00))
            x02 = (self.inrfLayer2.forward((x01)))
            x2 = self.pool(self.dp2d(x02))
            # CNN

            '''x00 = self.pool(F.relu(self.conv1((x))))
            x01 = self.pool(F.relu(self.conv2((x00))))'''

            x = x2.view(-1, 64 * 7 * 7)

            x = F.relu(self.fc1(x))
            x = self.dp(x)
            x = self.fc2(x)

            return x

    net = Net()
    net.load_state_dict(torch.load(filepath))
    net = net.to(device)
    net.eval()

    return net


def load_arc104_sin(device, filepath):
    # Architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.inrfLayer1 = INRF.INRF(dimG=1, dimW=5, numChanIn=1, numChanOut=32, sigma=5, paramSigma=0.01, lambdaV = 1.1)
            self.inrfLayer2 = INRF.INRF(dimG=1, dimW=5, numChanIn=32, numChanOut=64, sigma=5, paramSigma=0.01, lambdaV= 1.1)
            #1. / math.sqrt(1)

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(5 // 2, 5 // 2), bias=True)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(5 // 2, 5 // 2), bias=True)

            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 500)
            self.dp = nn.Dropout(p=0.5)
            self.dp2d = nn.Dropout2d(p=0.5)
            self.fc2 = nn.Linear(500, 10)

            #self.bn1 = nn.BatchNorm2d(3)
            #self.bn1 = nn.BatchNorm2d(3)

        def forward(self, x):

            #print(1. / math.sqrt(self.conv1.weight.size(1)))

            # I3N
            x00 =(self.inrfLayer1.forward((x)))
            x01 = self.pool((x00))
            x02 = (self.inrfLayer2.forward((x01)))
            x2 = self.pool(self.dp2d(x02))
            # CNN

            '''x00 = self.pool(F.relu(self.conv1((x))))
            x01 = self.pool(F.relu(self.conv2((x00))))'''

            x = x2.view(-1, 64 * 7 * 7)

            x = F.relu(self.fc1(x))
            x = self.dp(x)
            x = self.fc2(x)

            return x

    net = Net()
    net = net.to(device)

    net.load_state_dict(torch.load(filepath))
    net = net.to(device)
    net.eval()
    return net


def load_arc12_largekernels(device, filepath):
    # Architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.inrfLayer1 = INRF.INRF(dimG=1, dimW=15, numChanIn=1, numChanOut=2, sigma=1, paramSigma=10, lambdaV = 1.0)
            self.inrfLayer2 = INRF.INRF(dimG=1, dimW=11, numChanIn=2, numChanOut=3, sigma=1, paramSigma=10, lambdaV= 1.0)
            #1. / math.sqrt(1)

            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(3 * 7 * 7, 500)
            self.dp = nn.Dropout(p=0.5)
            self.dp2d = nn.Dropout2d(p=0.5)
            self.fc2 = nn.Linear(500, 10)

            #self.bn1 = nn.BatchNorm2d(3)
            #self.bn1 = nn.BatchNorm2d(3)

        def forward(self, x):

            #print(1. / math.sqrt(self.conv1.weight.size(1)))

            # I3N
            x00 =F.relu(self.inrfLayer1.forward((x)))
            x01 = self.pool((x00))
            x02 = F.relu(self.inrfLayer2.forward((x01)))
            x2 = self.pool(self.dp2d(x02))
            # CNN

            '''x00 = self.pool(F.relu(self.conv1((x))))
            x01 = self.pool(F.relu(self.conv2((x00))))'''

            x = x2.view(-1, 3 * 7 * 7)

            x = F.relu(self.fc1(x))
            x = self.dp(x)
            x = self.fc2(x)

            return x

    net = Net()
    net = net.to(device)

    net.load_state_dict(torch.load(filepath))
    net = net.to(device)
    net.eval()

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    wd = 0.00008 # weight decay
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=wd, amsgrad = True)

    return net, criterion, optimizer


def test_err_MNIST(net, batch_size, test_images, test_labels):

    # Test error
    print("-------------------------------------------")
    print("Test error full set")

    batchTest = 0
    testSize = 10000
    accCum = 0
    accCont = 0
    correct = 0
    total = 0
    while batchTest < testSize:
        net.eval()
        # batch_x, batch_y = mnist.train.next_batch(batch_size)

        if (batchTest + batch_size < testSize):

            batch_x = test_images[batchTest:batchTest + batch_size, :]
            batch_y = test_labels[batchTest:batchTest + batch_size]

            batchTest = batchTest + batch_size

        else:

            batch_x = test_images[batchTest:, :]
            batch_y = test_labels[batchTest:]
            batchTest = batchTest + batch_size

        inputs0 = np.moveaxis(batch_x, 2, 3)
        inputs0 = np.moveaxis(inputs0, 1, 2)

        inputs = torch.Tensor(inputs0).cuda()
        labels = torch.Tensor(batch_y).cuda()

        outputs,_,_ = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()


    accCum = correct / total
    print("Testing error:", 100 - 100 * accCum)
