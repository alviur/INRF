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
import dataLoader_CIFAR10

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('CIFAR10_images.png')
    #plt.show()


# Hardware
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128
lr=0.001
patience = 7 # used to modify the learning rate
patienceCont = 0
bestTest = 50

# Load CIFAR10 - https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
train_loader, valid_loader = dataLoader_CIFAR10.get_train_valid_loader("./",
                           batch_size,
                           augment=True,
                           random_seed = 47,
                           valid_size=0,
                           shuffle=True,
                           show_sample=False,
                           num_workers=1,
                           pin_memory=True,
                           normalizeF = True)

test_loader = dataLoader_CIFAR10.get_test_loader("./",
                    batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True,
                    normalizeF = True)


fileName = '/home/agomez/4_NCNN/1_algorithms/CIFAR10_CNN_ResNet20_SGD.pth'

# Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Input layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=(3 // 2, 3 // 2), bias=True)
        self.bn1 = nn.BatchNorm2d(16)

        # ==== 1st Stage =====
        # 1st Res block
        self.conv1_1_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=(3 // 2, 3 // 2), bias=True)
        self.bn1_1_1 = nn.BatchNorm2d(16)
        self.conv1_1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=(3 // 2, 3 // 2), bias=True)
        self.bn1_1_2 = nn.BatchNorm2d(16)

        # 2nd Res block
        self.conv1_2_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=(3 // 2, 3 // 2), bias=True)
        self.bn1_2_1 = nn.BatchNorm2d(16)
        self.conv1_2_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=(3 // 2, 3 // 2), bias=True)
        self.bn1_2_2 = nn.BatchNorm2d(16)

        # ==== 2nd Stage =====
        # 1st Res block
        self.conv2_1_1 = nn.Conv2d(in_channels=16, out_channels=16 * 2, kernel_size=3, padding=(3 // 2, 3 // 2), bias=True, stride = 2)
        self.bn2_1_1 = nn.BatchNorm2d(16*2)
        self.conv2_1_2 = nn.Conv2d(in_channels=16 * 2, out_channels=16 * 2, kernel_size=3, padding=(3 // 2, 3 // 2), bias=True)
        self.bn2_1_2 = nn.BatchNorm2d(16 * 2)
        self.conv2_1_3 = nn.Conv2d(in_channels=16, out_channels=16 * 2, kernel_size=1, bias=True, stride = 2)

        # 2nd Res block
        self.conv2_2_1 = nn.Conv2d(in_channels=16 * 2, out_channels=16 * 2, kernel_size=3, padding=(3 // 2, 3 // 2), bias=True)
        self.bn2_2_1 = nn.BatchNorm2d(16 * 2)
        self.conv2_2_2 = nn.Conv2d(in_channels=16 * 2, out_channels=16 * 2, kernel_size=3, padding=(3 // 2, 3 // 2), bias=True)
        self.bn2_2_2 = nn.BatchNorm2d(16 * 2)

        # ==== 3rd Stage =====
        # 1st Res block
        self.conv3_1_1 = nn.Conv2d(in_channels=16*2 , out_channels=16 * 4, kernel_size=3, padding=(3 // 2, 3 // 2), bias=True, stride = 2)
        self.bn3_1_1 = nn.BatchNorm2d(16 * 4)
        self.conv3_1_2 = nn.Conv2d(in_channels=16 * 4, out_channels=16 * 4, kernel_size=3, padding=(3 // 2, 3 // 2), bias=True)
        self.bn3_1_2 = nn.BatchNorm2d(16 * 4)
        self.conv3_1_3 = nn.Conv2d(in_channels=16 * 2, out_channels=16 * 4, kernel_size=1, bias=True, stride = 2)

        # 2nd Res block
        self.conv3_2_1 = nn.Conv2d(in_channels=16 * 4, out_channels=16 * 4, kernel_size=3, padding=(3 // 2, 3 // 2), bias=True)
        self.bn3_2_1 = nn.BatchNorm2d(16 * 4)
        self.conv3_2_2 = nn.Conv2d(in_channels=16 * 4, out_channels=16 * 4, kernel_size=3, padding=(3 // 2, 3 // 2), bias=True)
        self.bn3_2_2 = nn.BatchNorm2d(16 * 4)


        self.avgpool = nn.AvgPool2d(8,8)
        self.fc1 = nn.Linear(16*4, 10)

        # Initialize layers exactly as in Keras
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        torch.nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, x):

        #print(1. / math.sqrt(self.conv1.weight.size(1)))

        x00 = F.relu(self.bn1(self.conv1(x)))

        #Stage 1
        x1_1_1 = F.relu(self.bn1_1_1(self.conv1_1_1(x00)))
        x1_1_2 = F.relu(self.bn1_1_2(self.conv1_1_2(x1_1_1)))
        xres1_1 = F.relu(x00 + x1_1_2)
        x1_2_1 = F.relu(self.bn1_2_1(self.conv1_2_1(xres1_1)))
        x1_2_2 = F.relu(self.bn1_2_2(self.conv1_2_2(x1_2_1)))
        xres1_2 = F.relu(xres1_1 + x1_2_2)
        # Stage 2
        x2_1_1 = F.relu(self.bn2_1_1(self.conv2_1_1(xres1_2)))
        x2_1_2 = F.relu(self.bn2_1_2(self.conv2_1_2(x2_1_1)))
        x2_1_3 = self.conv2_1_3(xres1_2)
        xres2_1 = F.relu(x2_1_2 + x2_1_3)
        x2_2_1 = F.relu(self.bn2_2_1(self.conv2_2_1(xres2_1)))
        x2_2_2 = F.relu(self.bn2_2_2(self.conv2_2_2(x2_2_1)))
        xres2_2 = F.relu(x2_2_2 + xres2_1)
        # Stage 3
        x3_1_1 = F.relu(self.bn3_1_1(self.conv3_1_1(xres2_2)))
        x3_1_2 = F.relu(self.bn3_1_2(self.conv3_1_2(x3_1_1)))
        x3_1_3 = self.conv3_1_3(xres2_2)
        xres3_1 = F.relu(x3_1_2 + x3_1_3)
        x3_2_1 = F.relu(self.bn3_2_1(self.conv3_2_1(xres3_1)))
        x3_2_2 = F.relu(self.bn3_2_2(self.conv3_2_2(x3_2_1)))
        xres3_2 = F.relu(x3_2_2 + xres3_1)


        x04 = self.avgpool((xres3_2))
        x05 = x04.view(-1, 16*4 )
        x06 = self.fc1(x05)

        return x06

net = Net()
net = net.to(device)

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr

def lr_scheduleSGD(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    """
    lr = 0.1
    if epoch > 200:
        lr *= 0.001
    elif epoch > 135:
        lr *= 0.01
    elif epoch > 90:
        lr *= 0.1
    return lr

# Optimizer
criterion = nn.CrossEntropyLoss()
wd = 0.0003 # weight decay
optimizer = optim.SGD(net.parameters(), lr=0.0, momentum=0.9, weight_decay=wd)

#optimizer = optim.Adam(net.parameters(), lr=0.0, amsgrad = False)


epochs = 200
bestTest = 100
running_loss = 0
validationVerbosity = 5

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        net.train()

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_scheduleSGD(epoch)
            #param_group['lr'] = lr_schedule(epoch)

        # get the inputs; data is a list of [inputs, labels]
        inputs0, labels0 = data

        inputs = inputs0.cuda()
        labels = labels0.cuda()


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        # Kernel L2 regularizer
        l2_reg = None
        #reg_lambda = 1e-4
        reg_lambda = 0
        for W in net.parameters():
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)

        loss = criterion(outputs, labels) + l2_reg * reg_lambda
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
        for i, dataVal in enumerate(test_loader, 0):
            net.eval()
            # get the inputs; data is a list of [inputs, labels]
            inputs0, labels0 = dataVal

            inputs = inputs0.cuda()
            labels = labels0.cuda()

            # forward + backward + optimize
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            totalVal += labels.size(0)
            correctVal += (predicted == labels.long()).sum().item()

        accCum = correctVal / totalVal

        accCum = correctVal / totalVal
        print("val error:", 100 - 100 * accCum)

        if (bestTest < 100 - 100 * accCum):

            print("Best test Error:", bestTest)
        else:
            bestTest = 100 - 100 * accCum
            # Save model
            PATH = fileName
            torch.save(net.state_dict(), PATH)



# Test set
# Test error best validation
print("-------------------------------------------")
print("Test error best validation")
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
    outputs = net(inputs)

    _, predicted = torch.max(outputs.data, 1)
    totalVal += labels.size(0)
    correctVal += (predicted == labels.long()).sum().item()

accCum = correctVal / totalVal
print("-------------------------------")
print("Test error:", 100 - 100 * accCum)
print("-------------------------------")