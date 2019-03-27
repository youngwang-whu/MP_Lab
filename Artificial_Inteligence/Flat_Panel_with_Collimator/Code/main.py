# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Date: 02/18/2019
# Aim：
from __future__ import absolute_import, division, print_function
from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import Adam
from keras.preprocessing import sequence
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

"""
Notice: Modify "\\" into "/" in Linux server!
X = np.load("MP_Lab/Artificial_Inteligence/Flat_Panel_with_Collimator/data/collimator_H3_X.npy")
Y = np.load("MP_Lab/Artificial_Inteligence/Flat_Panel_with_Collimator/data/collimator_H3_Y.npy")
"""

X = np.load("MP_Lab\\Artificial_Inteligence\\Flat_Panel_with_Collimator\\data\\collimator_H3_X.npy")
Y = np.load("MP_Lab\\Artificial_Inteligence\\Flat_Panel_with_Collimator\\data\\collimator_H3_Y.npy")
X1 = X[:, 0:150:2]


# pre是在前填充， post是在后填充；
# dtype default: int32; dtype项务必要改成float！
X1 = sequence.pad_sequences(X1, maxlen=299, padding='post', value=0, dtype="float")
Y = sequence.pad_sequences(Y, maxlen=299, padding='post', value=0, dtype="float")
X2 = X[:, 1:150:2]
X3 = (X1 + X2)/2
x_train = X1[:, :, np.newaxis]
y_train = Y[:, :, np.newaxis]
# 插值
x1o = np.linspace(0, 75, 75)
x1n = np.linspace(0, 75, 299)
y1n = np.zeros((40, 299))
for i in range(40):
    f = interpolate.interp1d(x1o, X1[i, :], kind="cubic", )
    y1n[i, :] = f(x1n)


# Test Set (Based on the requirement of research)
x_test = x_train
y_test = y_train


# Create model.
model = Sequential()
model.add(Masking(mask_value=0, input_shape=(None, 1)))
model.add(LSTM(
        units=10,
        return_sequences=True,       # True: output at all steps. False: output as last step.
        stateful=False,              # True: the final state of batch1 is feed into the initial state of batch2
))
# add output layer
model.add(TimeDistributed(Dense(1)))
model.add(Masking(mask_value=0))
# Set Optimizer
opt = Adam(lr=0.001, decay=1e-6)
model.compile(optimizer=opt,
              loss='mse')
print(model.summary())
print('training---------------------------------')


hist = model.fit(x_train, y_train, verbose=1, validation_split=0.2, shuffle=True, nb_epoch=100)
print(hist.history)

score = model.evaluate(x_test, y_test, verbose=0)
print(model.metrics_names)
print('Test loss:', score)


# Save the model
model.save('rnn_model.h5')


# Input a 1D array.
def normalize(data):
    mx = max(data)
    mn = min(data)
    return [(float(i) - mn) / (mx-mn) for i in data]


# Plot figure
y_predict = model.predict(x_train)


def plot_predict_graph(x, y, y_predict, n, length=299):
    c = np.arange(0, length)
    y_predict = normalize(y_predict[n, :])
    plt.plot(c, y_predict, linestyle='--')
    plt.plot(c, y[n, :], linestyle='--')
    plt.plot(c, x[n, :], linestyle='--')
    plt.show()


model = load_model("collimator_lstm.h5")

