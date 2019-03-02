# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Date: 02/18/2019
# Aim：
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy import interpolate
import tensorflow as tf
import matplotlib.pyplot as plt
import xlrd


# Load validated data
ExcelFile = xlrd.open_workbook('C:\\Users\\wangya\\Documents\\Documents_wya\\MedPhysics\\collimator_H3.xlsx')
# Get the contents of sheet
P1 = ExcelFile.sheet_by_name('P1')
P3 = ExcelFile.sheet_by_name('P3')
P5 = ExcelFile.sheet_by_name('P5')
P7 = ExcelFile.sheet_by_name('P7')
P9 = ExcelFile.sheet_by_name('P9')


# Highlight:
# Solve the problem of appearing SPACE in an easy way of List index, after loading excel files.
# If you transform the list to numpy array directly without this process,
# each element in the array will have double quotation mark, thanks to the SPACE in the list.

depth_detected = np.array(P1.col_values(0)[1:151])
depth_dose = np.array(P1.col_values(9)[1:301])


# Input a 1D array.
def normalize(data):
    mx = max(data)
    mn = min(data)
    return [(float(i) - mn) / (mx-mn) for i in data]



def P_detect(Pi):
    x_old = np.linspace(1, 299, 150)
    x_new = np.linspace(1, 299, 299)
    P = np.empty((8, 299))
    for i in range(8):
        y = np.array(Pi.col_values(i+1)[1:151])
        f = interpolate.interp1d(x_old, y, kind='quadratic')
        P[i, :] = normalize(f(x_new))
    return P

P1_detect = P_detect(P1)
P3_detect = P_detect(P3)
P5_detect = P_detect(P5)
P7_detect = P_detect(P7)
P9_detect = P_detect(P9)



def P_dose(Pi):
    P = np.empty((8, 299))
    for i in range(10, 17):
        P[i-10, :] = normalize(np.array(Pi.col_values(i)[1:300]))
    return P

P1_dose = P_dose(P1)
P3_dose = P_dose(P3)
P5_dose = P_dose(P5)
P7_dose = P_dose(P7)
P9_dose = P_dose(P9)


# Training Set
X = np.concatenate((P1_detect, P3_detect, P5_detect, P7_detect, P9_detect), axis=0)
Y = np.concatenate((P1_dose, P3_dose, P5_dose, P7_dose, P9_dose), axis=0)
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

hist = model.fit(x_train, y_train, verbose=1, validation_split=0.2, shuffle=True)
print(hist.history)

score = model.evaluate(x_test, y_test, verbose=0)
print(model.metrics_names)
print('Test loss:', score)

# Plot figure of 3 situations
c = np.arange(0, 299)
y_pred = model.predict(x_test)
plt.plot(c, y_pred[1, :], linestyle='--')
plt.plot(c, y_test[1, :], linestyle='--')
plt.show()

# Performance
Model.Get_ModelLoss_figure(hist)

# Save the model
model.save('rnn_model.h5')


class (object):

    def __init__(self, depth, energy):
        self.depth = depth
        self.energy = energy

    def

