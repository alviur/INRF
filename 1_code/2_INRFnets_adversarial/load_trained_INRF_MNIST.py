#!/usr/bin/env python
# Libraries

import numpy as np
import matplotlib.pyplot as plt
import INRF_loop as INRF
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import load_data
import math

def load_arc_043_ReLu(device, filepath):
    # Architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.inrfLayer1 = INRF.INRF(dimG=1, dimW=5, numChanIn=1, numChanOut=32, sigma=1, paramSigma=100, lambdaV = 1.1)
            self.inrfLayer2 = INRF.INRF(dimG=1, dimW=5, numChanIn=32, numChanOut=64, sigma=1, paramSigma=100, lambdaV= 1.1)

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(5 // 2, 5 // 2), bias=False)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(5 // 2, 5 // 2), bias=False)

            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 500)
            self.dp = nn.Dropout(p=0.5)
            self.dp2d = nn.Dropout2d(p=0.5)
            self.fc2 = nn.Linear(500, 10)


        def forward(self, x):
            # INRF
            x00 =(self.inrfLayer1.forward((x)))
            x01 = self.pool((x00))
            x02 = (self.inrfLayer2.forward((x01)))
            x2 = self.pool(self.dp2d(x02))

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

