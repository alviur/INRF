#############################################################
#
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
import load_data
import math

# Paths
pathSaveFM = '/home/agomez/4_NCNN/1_algorithms/sampleOuts/'

# Hardware
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
lr=0.001
patience = 7 # used to modify the learning rate
patienceCont = 0
bestTest = 50

# Data
#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

# Load dataset
train_labels,train_labels0\
    ,test_labels,val_labels\
    ,train_images,val_images\
    ,test_images,train_images0 = load_data.loadMnist()

fileName = '/home/agomez/4_NCNN/1_algorithms/loop_arc6_wi2_avgP_32_32_128_allR_005.pth'

# Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inrfLayer1 = INRF.INRF(dimG=1, dimW=5, numChanIn=1, numChanOut=32, sigma=1, paramSigma=0.05, lambdaV = 1)
        self.inrfLayer2 = INRF.INRF(dimG=1, dimW=5, numChanIn=32, numChanOut=32, sigma=1, paramSigma=0.01, lambdaV= 1)
        self.inrfLayer3 = INRF.INRF(dimG=1, dimW=5, numChanIn=32, numChanOut=128, sigma=1, paramSigma=0.01, lambdaV= 1)
        #1. / math.sqrt(1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, padding=(5 // 2, 5 // 2), bias=False)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=(5 // 2, 5 // 2), bias=False)

        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.avg = nn.AvgPool2d(7, 7)
        self.fc1 = nn.Linear(64 * 4 * 4, 100)
        self.dp = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)

        #self.bn1 = nn.BatchNorm2d(3)
        #self.bn1 = nn.BatchNorm2d(3)

    def forward(self, x):

        #print(1. / math.sqrt(self.conv1.weight.size(1)))

        # I3N
        x00 =(self.inrfLayer1.forward((x)))
        x01 = (self.inrfLayer2.forward((x00)))
        x1 = self.pool(x01)
        x02 = (self.inrfLayer3.forward((x1)))
        x2 = self.pool2(x02)
        x3 = self.avg(x2)
        # CNN

        '''x00 = self.pool(F.relu(self.conv1((x))))
        x01 = self.pool(F.relu(self.conv2((x00))))'''

        x = x3.view(-1, 128)
        x = self.fc2(x)

        return x,x00,x01


net = Net()
net = net.to(device)
#net.cuda()


# Optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)


training_iters = 4000000
display_step = 50
step = 1
bestTest = 100
batchCont = 0
patienceCont1 = 0
# Keep training until reach max iterations
running_loss = 0
test_step = 50000
while step * batch_size < training_iters:
    net.train()
    # batch_x, batch_y = mnist.train.next_batch(batch_size)

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

    # Run optimization op (backprop)
    # zero the parameter gradients
    optimizer.zero_grad()

    #with torch.autograd.detect_anomaly():
    # forward + backward + optimize
    outputs, fm, fm2 = net(inputs)

    #print(labels.long())
    loss = criterion(outputs, labels.long())
    loss.backward()
    optimizer.step()

    if step==1:
        '''print(torch.sum(torch.isnan(fm)),torch.sum(torch.isnan(fm2)),
              torch.sum(torch.isinf(fm)),torch.sum(torch.isinf(fm2)), "=====INRF1======",step)
        print(torch.sum(torch.isnan(net.inrfLayer1.w.weight.grad)),
              torch.sum(torch.isinf(net.inrfLayer1.w.weight.grad)), "Gradients! INRF1!")
        print(torch.sum(torch.isnan(net.inrfLayer2.w.weight.grad)),
              torch.sum(torch.isinf(net.inrfLayer2.w.weight.grad)), "Gradients! INRF2!")
        print(net.inrfLayer2.w.weight.grad)
        print(net.inrfLayer1.w.weight.grad)'''




    # print statistics
    running_loss += loss.item()


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

            # Save some outputs
            #np.save(pathSaveFM+'feature_maps_'+str(batchTest),fm.cpu())
            #np.save(pathSaveFM + 'inputs_' + str(batchTest), inputs.cpu())

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

# Test error
print("-------------------------------------------")
print("Test error before full set")
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


# Train with whole training set
extraEpochs = 0
display_step = 50
step = 1
bestTest = 100
batchCont = 0
patienceCont1 = 0

running_loss = 0
test_step = 50000
while extraEpochs < 3:
    net.train()
    # batch_x, batch_y = mnist.train.next_batch(batch_size)

    if (batchCont + batch_size < 60000):

        batch_x = train_images0[batchCont:batchCont + batch_size, :]
        batch_y = train_labels0[batchCont:batchCont + batch_size].astype(int)

        batchCont = batchCont + batch_size

    else:

        batch_x = train_images0[batchCont:60000, :]
        batch_y = train_labels0[batchCont:60000].astype(int)

        batchCont = 0
        extraEpochs += 1

    inputs0 = np.moveaxis(batch_x, 2, 3)
    inputs0 = np.moveaxis(inputs0, 1, 2)

    inputs = torch.Tensor(inputs0).cuda()
    labels = torch.Tensor(batch_y).cuda()

    # Run optimization op (backprop)
    # zero the parameter gradients
    optimizer.zero_grad()

    #with torch.autograd.detect_anomaly():
    # forward + backward + optimize
    outputs, fm, fm2 = net(inputs)

    #print(labels.long())
    loss = criterion(outputs, labels.long())
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()


    if step % display_step == 0:
        print("Step:",step * batch_size," Loss:",running_loss / display_step)
        running_loss = 0.0

    step += 1

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
