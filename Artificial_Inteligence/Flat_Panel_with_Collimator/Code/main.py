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


X = np.load("collimator_H3_X.npy")
Y = np.load("collimator_H3_Y.npy")
X1 = X[:, 0:150:2]
X2 = X[:, 1:150:2]
X3 = (X1 + X2)/2
x_train = X[:, :, np.newaxis]
y_train = Y[:, :, np.newaxis]


# pre是在前填充， post是在后填充；
x_train = sequence.pad_sequences(x_train, maxlen=299, padding='post', value=-1)


# Test Set (Based on the requirement of research)

x_test = x_train
y_test = y_train


# Create model.
model = Sequential()
model.add(Masking(mask_value=-1, input_shape=(299, 1)))
model.add(LSTM(
        units=10,
        return_sequences=True,       # True: output at all steps. False: output as last step.
        stateful=False,              # True: the final state of batch1 is feed into the initial state of batch2
))
# add output layer
model.add(TimeDistributed(Dense(1)))
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


# Performance
model.Get_ModelLoss_figure(hist)

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

model = load_model("C:\\Users\\wangya\\Desktop\\collimator_lstm.h5")

