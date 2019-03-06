# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Date: 02/18/2019
# Aim：
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


X = np.load("collimator_H3_X.npy")
Y = np.load("collimator_H3_Y.npy")
x_train = X[:, :, np.newaxis]
y_train = Y[:, :, np.newaxis]


# Test Set (Based on the requirement of research)

x_test = x_train
y_test = y_train


model = tf.keras.Sequential()
# build a LSTM layer
model.add(tf.keras.layers.LSTM(
    input_shape=(299, 1),
    units=10,
    return_sequences=True,       # True: output at all steps. False: output as last step.
    stateful=False,              # True: the final state of batch1 is feed into the initial state of batch2
))
# add output layer
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
adam = tf.keras.optimizers.Adam(0.006)
model.compile(optimizer=adam,
              loss='mse')
print(model.summary())        
print('training---------------------------------')

hist = model.fit(x_train, y_train, verbose=1, validation_split=0.2, shuffle=True, nb_epoch=1000)
print(hist.history)

score = model.evaluate(x_test, y_test, verbose=0)
print(model.metrics_names)
print('Test loss:', score)

# Plot figure of 3 situations
c = np.arange(0, 299)
y_pred = model.predict(x_test)
plt.plot(c, y_pred[1, :], linestyle='--')
plt.plot(c, y_test[1, :], linestyle='--')
plt.plot(c, x_test[1, :], linestyle='--')
plt.show()


# Performance
model.Get_ModelLoss_figure(hist)

# Save the model
model.save('rnn_model.h5')

            
