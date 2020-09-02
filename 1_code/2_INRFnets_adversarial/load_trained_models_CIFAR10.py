#############################################################
#
#
#############################################################

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import INRF_loop_staticpq as INRF
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


def load_cnnCIFAR10(device, filepath):

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.inrfLayer1 = INRF.INRF(dimG=1, dimW=5, numChanIn=3, numChanOut=192, sigma=1, paramSigma=0.08, lambdaV = 2.0)
            self.inrfLayer2 = INRF.INRF(dimG=1, dimW=5, numChanIn=192, numChanOut=192, sigma=1, paramSigma=0.08, lambdaV= 2.0)
            self.inrfLayer3 = INRF.INRF(dimG=1, dimW=5, numChanIn=192, numChanOut=192, sigma=1, paramSigma=0.08, lambdaV=2.0)
            #1. / math.sqrt(1)

            self.conv1 = nn.Conv2d(in_channels=3, out_channels=192, kernel_size=5, padding=(5 // 2, 5 // 2), bias=True)
            self.conv2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=(5 // 2, 5 // 2), bias=True)
            self.conv3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=(5 // 2, 5 // 2), bias=True)

            self.pool = nn.MaxPool2d(2, 2)
            self.fc3 = nn.Linear(192, 10)

            self.bn1 = nn.BatchNorm2d(192)
            self.bn2 = nn.BatchNorm2d(192)
            self.bn3 = nn.BatchNorm2d(192)

            self.avgpool = nn.AvgPool2d(8,8)

        def forward(self, x):
            # CNN

            x00 = self.pool((self.bn1(F.relu(self.conv1((x))))))
            x01 = self.bn2(F.relu(self.conv2((x00))))
            x03 = self.pool(x01)
            x04 = self.bn3(self.conv3((x03)))
            x05 = self.avgpool((x04))


            x = x05.view(-1, 192 )
            x = self.fc3(x)

            return x

    net = Net()
    net.load_state_dict(torch.load(filepath))
    net = net.to(device)
    net.eval()

    return net

def load_arc16_7(device, filepath):

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.inrfLayer1 = INRF.INRF(dimG=1, dimW=5, numChanIn=3, numChanOut=192, sigma=3, paramSigma=0.08, lambdaV = 2.0)
            self.inrfLayer2 = INRF.INRF(dimG=1, dimW=5, numChanIn=192, numChanOut=192, sigma=3, paramSigma=0.08, lambdaV= 2.0)
            self.inrfLayer3 = INRF.INRF(dimG=1, dimW=5, numChanIn=192, numChanOut=192, sigma=3, paramSigma=0.08, lambdaV=2.0)
            #1. / math.sqrt(1)

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(5 // 2, 5 // 2), bias=True)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(5 // 2, 5 // 2), bias=True)

            self.pool = nn.MaxPool2d(2, 2)
            self.fc3 = nn.Linear(192, 10)

            self.bn1 = nn.BatchNorm2d(192)
            self.bn2 = nn.BatchNorm2d(192)
            self.bn3 = nn.BatchNorm2d(192)

            self.avgpool = nn.AvgPool2d(8,8)

        def forward(self, x):

            x00 = self.pool(self.bn1(F.relu(self.inrfLayer1.forward((x)))))
            x01 = self.bn2(F.relu(self.inrfLayer2.forward((x00))))
            x03 = self.pool(x01)
            x04 = self.bn3(F.relu(self.inrfLayer3.forward((x03))))
            x05 = self.avgpool((x04))

            x = x05.view(-1, 192 )
            x = self.fc3(x)

            return x

    net = Net()
    net.load_state_dict(torch.load(filepath))
    net = net.to(device)
    net.eval()

    return net

