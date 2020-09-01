#############################################################
# Training code for CIFAR-10 dataset
#############################################################

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import INRF2_loop as INRF
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import dataLoader_CIFAR10
import time


start = time.time()

# Hardware
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
lr=0.001
patience = 7 # used to modify the learning rate
patienceCont = 0
bestTest = 50

# Load CIFAR10
train_loader, valid_loader = dataLoader_CIFAR10.get_train_valid_loader("./",
                           batch_size,
                           augment=False,
                           random_seed = 47,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=1,
                           pin_memory=True,
                           normalizeF = False)

test_loader = dataLoader_CIFAR10.get_test_loader("./",
                    batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True,
                    normalizeF = False)

# Path to save model
fileName = '/home/agomez/4_NCNN/1_algorithms/loop_arc2Cifar10_008_dp2_dp3.pth'

# Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inrfLayer1 = INRF.INRF(dimG=1, dimW=5, numChanIn=3, numChanOut=192, sigma=1, paramSigma=0.08, lambdaV = 2.0)
        self.inrfLayer2 = INRF.INRF(dimG=1, dimW=5, numChanIn=192, numChanOut=192, sigma=1, paramSigma=0.08, lambdaV= 2.0)
        self.inrfLayer3 = INRF.INRF(dimG=1, dimW=5, numChanIn=192, numChanOut=192, sigma=1, paramSigma=0.08, lambdaV=2.0)

        '''# Uncomment for CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=192, kernel_size=5, padding=(5 // 2, 5 // 2), bias=True)
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=(5 // 2, 5 // 2), bias=True)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=(5 // 2, 5 // 2), bias=True)
        '''

        self.pool = nn.MaxPool2d(2, 2)
        self.fc3 = nn.Linear(192, 10)

        self.dp2d1 = nn.Dropout2d(p=0.0)
        self.dp2d2 = nn.Dropout2d(p=0.25)
        self.dp2d3 = nn.Dropout2d(p=0.25)

        self.avgpool = nn.AvgPool2d(8,8)

    def forward(self, x):
        # I3N
        x00 = self.pool((self.inrfLayer1.forward((x))))
        x01 = self.dp2d2(self.inrfLayer2.forward((x00)))
        x03 = self.pool(x01)
        x04 = self.dp2d3(self.inrfLayer3.forward((x03)))
        x05 = self.avgpool((x04))

        # CNN
        '''# Uncomment for CNN
        x00 = self.pool(F.relu(self.conv1((x))))
        x01 = self.pool(F.relu(self.conv2((x00))))
        x2 =x01'''

        x = x05.view(-1, 192 )
        x = self.fc3(x)

        return x,x00,x01


net = Net()
net = net.to(device)
#net.cuda()


# Optimizer
criterion = nn.CrossEntropyLoss()
wd = 0.0 # weight decay
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=wd, amsgrad = True)


epochs = 200 # Number of epochs
bestTest = 100
running_loss = 0
validationVerbosity = 5

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        net.train()
        # get the inputs; data is a list of [inputs, labels]
        inputs0, labels0 = data
        inputs = inputs0.cuda()
        labels = labels0.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs,_,_ = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        for group in optimizer.param_groups:
            for param in group['params']:
                param.data = param.data.add(-wd * group['lr'], param.data)
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()

    accCum = correct / total
    print("Epoch:",epoch,"- Loss:", running_loss / 703, " - Training error:",100 - 100 * accCum)
    running_loss = 0.0
    correct = 0
    total = 0

    if epoch % validationVerbosity ==0:

        correctVal = 0
        totalVal = 0
        for i, dataVal in enumerate(valid_loader, 0):
            net.eval()
            # get the inputs; data is a list of [inputs, labels]
            inputs0, labels0 = dataVal

            inputs = inputs0.cuda()
            labels = labels0.cuda()

            # forward + backward + optimize
            outputs, _, _ = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            totalVal += labels.size(0)
            correctVal += (predicted == labels.long()).sum().item()

        accCum = correctVal / totalVal
        print("Validation error:", 100 - 100 * accCum)

        if (bestTest < 100 - 100 * accCum):

            print("Best validation Error:", bestTest)
        else:
            bestTest = 100 - 100 * accCum
            # Save model
            PATH = fileName
            torch.save(net.state_dict(), PATH)


end = time.time()
print("Trainning time", end - start)

# Test set
start = time.time()

correctVal = 0
totalVal = 0
for i, dataVal in enumerate(test_loader, 0):
    net.eval()
    # get the inputs; data is a list of [inputs, labels]
    inputs0, labels0 = dataVal

    inputs = inputs0.cuda()
    labels = labels0.cuda()

    # forward + backward + optimize
    outputs, _, _ = net(inputs)

    end = time.time()
    print("Forward time", end - start)

    _, predicted = torch.max(outputs.data, 1)
    totalVal += labels.size(0)
    correctVal += (predicted == labels.long()).sum().item()

accCum = correctVal / totalVal
print("-------------------------------")
print("Test error en of training:", 100 - 100 * accCum)
print("-------------------------------")

# Test error best validation
print("-------------------------------------------")
print("Test error best validation accuracy")
print("-------------------------------------------")

PATH = fileName
net = Net()
net.load_state_dict(torch.load(PATH))
net = net.to(device)
net.eval()
correctVal = 0
totalVal = 0
for i, dataVal in enumerate(test_loader, 0):
    net.eval()
    # get the inputs; data is a list of [inputs, labels]
    inputs0, labels0 = dataVal

    inputs = inputs0.cuda()
    labels = labels0.cuda()

    # forward + backward + optimize
    outputs, _, _ = net(inputs)

    _, predicted = torch.max(outputs.data, 1)
    totalVal += labels.size(0)
    correctVal += (predicted == labels.long()).sum().item()

accCum = correctVal / totalVal
print("-------------------------------")
print("Test error:", 100 - 100 * accCum)
print("-------------------------------")


