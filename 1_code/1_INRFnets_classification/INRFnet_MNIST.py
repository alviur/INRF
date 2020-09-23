#############################################################
# Deeply-supervised networks architecture
#
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
import dataLoader_MNIST
import math

# Paths
pathSaveFM = '/home/agomez/4_NCNN/1_algorithms/sampleOuts/'

# Hardware
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load dataset, feel free to change this for Pytorch version of MNIST dataset
train_labels,train_labels0\
    ,test_labels,val_labels\
    ,train_images,val_images\
    ,test_images,train_images0 = dataLoader_MNIST.loadMnist2()

# Path to save model
fileName = '/home/agomez/4_NCNN/1_algorithms/MNIST.pth'

# Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inrfLayer1 = INRF.INRF(dimG=1, dimW=5, numChanIn=1, numChanOut=32, sigma=1, paramSigma=100, lambdaV = 1.1)
        self.inrfLayer2 = INRF.INRF(dimG=1, dimW=5, numChanIn=32, numChanOut=64, sigma=1, paramSigma=100, lambdaV= 1.1)

        '''# Uncomment for CNN equivalent
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(5 // 2, 5 // 2), bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(5 // 2, 5 // 2), bias=False)'''

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 500)
        self.dp = nn.Dropout(p=0.5)
        self.dp2d = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(500, 10)



    def forward(self, x):


        # I3N
        x00 =(self.inrfLayer1.forward((x)))
        x01 = self.pool((x00))
        x02 = (self.inrfLayer2.forward((x01)))
        x2 = self.pool(self.dp2d(x02))

        # Uncomment for CNN equivalent
        '''x00 = self.pool(F.relu(self.conv1((x))))
        x01 = self.pool(F.relu(self.conv2((x00))))'''

        x = x2.view(-1, 64 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.dp(x)
        x = self.fc2(x)

        return x,x00,x01


net = Net()
net = net.to(device)


# Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Keep training until reach max iterations
training_iters = 4000000
display_step = 50
# Loop variables
batch_size = 64
step = 1
bestTest = 100
batchCont = 0
patience = 7 # used to modify the learning rate
patienceCont = 0
bestTest = 50


running_loss = 0
test_step = 50000

# Training loop
while step * batch_size < training_iters:
    net.train()

    if (batchCont + batch_size < 50000):

        batch_x = train_images[batchCont:batchCont + batch_size, :]
        batch_y = train_labels[batchCont:batchCont + batch_size].astype(int)

        batchCont = batchCont + batch_size

    else:

        batch_x = train_images[batchCont:50000, :]
        batch_y = train_labels[batchCont:50000].astype(int)

        batchCont = 0

    inputs0 = np.moveaxis(batch_x, 2, 3)
    inputs0 = np.moveaxis(inputs0, 1, 2)

    inputs = torch.Tensor(inputs0).cuda()
    labels = torch.Tensor(batch_y).cuda()


    # zero the parameter gradients
    optimizer.zero_grad()

    outputs, fm, fm2 = net(inputs)

    loss = criterion(outputs, labels.long())
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    # Print statistics
    if step % display_step == 0:
        print("Step:",step * batch_size," Loss:",running_loss / display_step)
        running_loss = 0.0

    step += 1

    if step * batch_size > test_step:

        test_step += 50000

        # Validation error
        batchTest = 0
        testSize = 10000
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

            inputs0 = np.moveaxis(batch_x, 2, 3)
            inputs0 = np.moveaxis(inputs0, 1, 2)

            inputs = torch.Tensor(inputs0).cuda()
            labels = torch.Tensor(batch_y).cuda()

            outputs, fm, fm2 = net(inputs)


            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()

        accCum = correct / total

        if (bestTest < 100 - 100 * accCum):

            print("Best validation Error:", bestTest)

        else:
            print("validation Error:", 100 - 100 * accCum)
            bestTest = 100 - 100 * accCum
            patienceCont = 0

            # Save model
            PATH = fileName
            torch.save(net.state_dict(), PATH)

        if (patienceCont == patience):
            lr = lr / 2

        if (bestTest > 100- accCum*100):
            bestTest = accCum
            patienceCont = 0
        else:
            patienceCont += 1

        if (patienceCont == patience):
            lr = lr / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


# Test error best validation
print("-------------------------------------------")

PATH = fileName
net = Net()
net.load_state_dict(torch.load(PATH))
net = net.to(device)
net.eval()

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

