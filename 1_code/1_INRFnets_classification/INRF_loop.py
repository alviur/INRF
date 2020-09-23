##########################################
# Implementation of INRF layer on Pytorch
#
##########################################
# Inputs:
#   -dimG: dimension kernel g, not implemented yet, ignore
#   -dimW: dimension of kernel w
#   -numChanIn: dimension of input channels
#   -numChanOut: dimension of output channels
#   -sigma: Nonlinearity used:
#        1: Relu
#        2: Tanh
#        3: pq sigmoid with p=0.7 and q=0.3
#        4: pq sigmoid with p and q trainable
#        5: Sinusoid activation function
#   -paramSigma: Normalization factor, initial weight values will be divided by this
#   -lambdaV: weight of the nonlinearity part
#   -stride: not implemented yet, ignore

# Libraries
import torch
import numpy as np
import torch.nn as nn

class INRF(nn.Module):

    def __init__(self, dimG=1, dimW=3, numChanIn=3, numChanOut=3, sigma= 1, paramSigma= 12, lambdaV = 2.0, stride=1):

        super(INRF, self).__init__()
        self.g = torch.nn.Conv2d(numChanIn, numChanOut, dimG, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.w = nn.Parameter(torch.randn(dimW*dimW, numChanOut, numChanIn, 1, 1,requires_grad=True)/paramSigma)
        self.stride = stride
        self.lamda = lambdaV
        self.pandQ = nn.Parameter(torch.autograd.Variable(torch.from_numpy(np.asarray([0.5,0.5])), requires_grad=True).type(torch.FloatTensor))
        self.paramSigma = paramSigma
        self.numChanIn = numChanIn
        self.dimW = dimW
        # Create difference masks
        self.diffMasks = nn.Parameter(self.createMask(dimW, numChanIn), requires_grad=False).cuda()
        self.sigma = sigma

    def forward(self, x):

        for i in range(self.dimW * self.dimW):

            pad = self.dimW // 2
            # Compute differences
            convw = torch.nn.functional.conv2d(x, self.diffMasks[i, :, :, :].unsqueeze(1),
                                                padding=(pad, pad),
                                                groups=self.numChanIn)

            # Non-linearity
            sub = convw +  self.lamda*self.activation(x, convw, self.sigma)

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
