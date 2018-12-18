# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Date: 2018.12.13
# Aim：
from __future__ import absolute_import, division, print_function

import xlrd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Add noise
X = np.load("x.npy")
Y = np.load("y.npy")


def Activity_add_noise(Inputsize, Inputlength, SNR):
    x_add = X*(np.ones((Inputsize, Inputlength)) + np.random.randn(Inputsize, Inputlength)/SNR)
    return x_add


X = Activity_add_noise(200, 302, 10)
x_train = X[:, :, np.newaxis]
y_train = Y[:, :, np.newaxis]

# Train with new data
model = tf.keras.models.load_model('rnn_model.h5')
hist = model.fit(x_train, y_train, batch_size=80, epochs=200, verbose=1, validation_split=0.2, shuffle=True)
print(hist.history)

model.save('rnn_model_0903.h5')

