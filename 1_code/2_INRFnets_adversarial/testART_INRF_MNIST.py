import sys, os

from load_trained_INRF_MNIST import load_arc_043_ReLu

# Imports PyTorch, Numpy, etc
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# Imports ART
from art.attacks import FastGradientMethod, DeepFool, CarliniL2Method, CarliniLInfMethod
from art.classifiers import PyTorchClassifier
from art.utils import load_mnist

filepath = '../../2_models/MNIST_0.429.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = load_arc_043_ReLu(device, filepath)
# Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# Load MNIST data for ART models
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
# Swap axes to PyTorch's NCHW format
x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

# Create the ART classifier

classifier = PyTorchClassifier(
    model=net,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

# Evaluate the ART classifier on benign test examples
x_test = np.swapaxes(x_test, 2, 3)
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Begin FGSM Attack
for epsilon in np.arange(0.1, 0.6, 0.1):
    t = time.time()
    # Generate adversarial test examples
    attack = FastGradientMethod(classifier=classifier, eps=epsilon)
    x_test_adv = attack.generate(x=x_test)
    print(time.time() - t)
    # Evaluate the ART classifier on adversarial test examples

    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial test examples for eps={}: {}".format(epsilon, accuracy * 100))
# End FGSM Attack

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
# End DeepFool Attack

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
