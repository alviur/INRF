#!/usr/bin/env python
# Libraries

import numpy as np
import matplotlib.pyplot as plt
import INRF_loop_staticpq as INRF
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

