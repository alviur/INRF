##########################################
# Implementation of INRF layer on Pytorch
#
##########################################
# Attributes:
#   -dimG:
#   -dimW:
#   -numChanIn:
#   -numChanOut:
#   -sigma:
#   -paramSigma:

# Libraries
import torch
import numpy as np
import torch.nn as nn

class INRF(nn.Module):

    def __init__(self, dimG=1, dimW=3, numChanIn=3, numChanOut=3, sigma= 'relu', paramSigma= 0.6, lambdaV = 2.0, stride=1):

        super(INRF, self).__init__()

        self.g = torch.nn.Conv2d(numChanIn, numChanOut, dimG, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.w = nn.Parameter(torch.randn(dimW*dimW, numChanOut, numChanIn, 1, 1,requires_grad=True)/paramSigma)

        self.stride = stride
        self.lamda = lambdaV
        self.pandQ = nn.Parameter(torch.empty(2).normal_(mean=0.5, std=0.5)).cuda()

        self.paramSigma = paramSigma
        self.numChanIn = numChanIn
        self.dimW = dimW
        self.diffMasks = nn.Parameter(self.createMask(dimW, numChanIn), requires_grad=False).cuda()
        self.sigma = sigma

    def forward(self, x):

        # print(self.diffMasks["wl1"].shape)

        for i in range(self.dimW * self.dimW):

            pad = self.dimW // 2
            convw = torch.nn.functional.conv2d(x, self.diffMasks[i, :, :, :].unsqueeze(1),
                                                padding=(pad, pad),
                                                groups=self.numChanIn)

            # Non-linearity

            sub = convw + self.lamda*self.activation(x, convw, self.sigma)

            if i == 0:
                w = torch.nn.functional.conv2d(sub, self.w[i, :, :, :], stride=self.stride)
            else:
                w += torch.nn.functional.conv2d(sub, self.w[i, :, :, :], stride=self.stride)

        return w

    def activation(self, x, convx, option):

        if(option==1):
            return torch.nn.functional.relu(x-convx)

        elif(option==2):
            return torch.nn.functional.tanh(x-convx)

        elif (option == 3):

            signs = torch.sign(x - convx)
            signsNegative = torch.sign(x - convx)*(-1)
            diff = torch.nn.functional.relu(x - convx)
            diff2 = torch.nn.functional.relu((x - convx)*signsNegative)

            positivePart = diff**0.8
            negativePart = diff2 ** 0.3

            out = (positivePart +  negativePart)*signs

            return out

        elif (option == 4):

            signs = torch.sign(x - convx)
            signsNegative = torch.sign(x - convx) * (-1)
            diff = torch.nn.functional.relu(x - convx)
            diff2 = torch.nn.functional.relu((x - convx) * signsNegative)

            positivePart = diff ** self.pandQ[0]
            negativePart = diff2 ** self.pandQ[1]

            out = (positivePart + negativePart) * signs

            return out

        elif (option == 5):
            return torch.sin(x - convx)


    def createMask(self, dimW, numChanIn):

        n = numChanIn

        # Create empty zero matrix
        diffMasksVect = np.zeros((dimW * dimW, numChanIn, dimW, dimW))

        # Counters for rows and cols
        contRow = 0
        contCols = 0

        for i in range((dimW * dimW)):



            w = np.zeros((dimW,dimW))
            w[contRow,contCols] = 1
            contCols += 1
            if ( contCols>=dimW):
                contRow += 1
                contCols = 0

            diffMasksVect[i, :, :, :] = np.stack([w] * n, axis=0)

        return torch.Tensor(diffMasksVect)
