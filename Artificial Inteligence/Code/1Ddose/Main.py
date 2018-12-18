# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Date: 2018.12.12
# Aim：
from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt
import Model


# Training Set
X = np.load("x.npy")
Y = np.load("y.npy")
x_train = X[:, :, np.newaxis]
y_train = Y[:, :, np.newaxis]


# Test Set (Based on the requirement of research)
## For example:
x = X[0:99:4, :]
y = Y[0:99:4, :]
x_test = x[:, :, np.newaxis]
y_test = y[:, :, np.newaxis]

# Train the model
model, hist, score = Model.train_model_lstm(x_train, y_train, x_test, y_test, 12, 80, 200)

# Performance
Model.Get_ModelLoss_figure(hist)

# Save the model
model.save('rnn_model.h5')
