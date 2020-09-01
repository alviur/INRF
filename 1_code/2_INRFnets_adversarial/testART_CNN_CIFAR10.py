import sys, os

from load_trained_models_CIFAR10 import load_cnnCIFAR10

# Imports PyTorch, Numpy, etc
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# Imports ART
from art.attacks import FastGradientMethod
from art.classifiers import PyTorchClassifier
from art.utils import load_dataset

filepath = '../../2_models/CIFAR10_CNN_24.28.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = load_cnnCIFAR10(device, filepath)

# Optimizer
criterion = nn.CrossEntropyLoss()
wd = 0.00008 # weight decay
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=wd, amsgrad = True)

# Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("cifar10"))
x_test = np.float32(np.transpose(x_test, (0, 3, 1, 2)))
im_shape = x_train[0].shape

# Create the ART classifier

classifier = PyTorchClassifier(
    model=net,
    clip_values=(0, 1),
    loss=criterion,
    optimizer=optimizer,
    input_shape=im_shape,
    nb_classes=10,
)

# Evaluate the ART classifier on benign test examples
# x_test = np.swapaxes(x_test, 2, 3)
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Begin FGSM
for epsilon in np.arange(0.02, 0.16, 0.02):
    t = time.time()
    # Generate adversarial test examples
    attack = FastGradientMethod(classifier=classifier, eps=epsilon)
    x_test_adv = attack.generate(x=x_test)
    # print(time.time() - t)
    # Evaluate the ART classifier on adversarial test examples

    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial test examples for eps={}: {}".format(epsilon, accuracy * 100))
# end FGSM

# Begin DeepFool Attack
t = time.time()
# Generate adversarial test examples
attack = DeepFool(classifier=classifier)
x_test_adv = attack.generate(x=x_test)
print(time.time() - t)

# Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples for DeepFool: {} %".format(accuracy * 100))
# End DeepFool

# Begin Carlini-L2
t = time.time()
# Generate adversarial test examples
attack = CarliniL2Method(classifier=classifier)
x_test_adv = attack.generate(x=x_test)
print(time.time() - t)

# Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples for Carlini-L2: {} %".format(accuracy * 100))
# End Carlini-L2

# Begin Carlini-Linf
t = time.time()
# Generate adversarial test examples
attack = CarliniLInfMethod(classifier=classifier)
x_test_adv = attack.generate(x=x_test)
print(time.time() - t)

# Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples for Carlini-Linf: {} %".format(accuracy * 100))
# End Carlini-Linf
