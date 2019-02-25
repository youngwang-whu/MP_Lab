# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Date: 01/25/2019
# Aim：Train LSTM model using 1D PET as training set
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import xlrd


# Load validated data
ExcelFile = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\Validation_data.xlsx')
# Get the contents of sheet
sheet = ExcelFile.sheet_by_name('Sheet1')

# Highlight:
# Solve the problem of appearing SPACE in an easy way of List index, after loading excel files.
# If you transform the list to numpy array directly without this process,
# each element in the array will have double quotation mark, thanks to the SPACE in the list.
cols1 = np.array(sheet.col_values(1)[1:])
cols2 = np.array(sheet.col_values(2)[1:])
cols3 = np.array(sheet.col_values(3)[1:])
cols4 = np.array(sheet.col_values(4)[1:])
cols5 = np.array(sheet.col_values(5)[1:])
cols6 = np.array(sheet.col_values(6)[1:])
# To transpose the 1D numpy array.
cols1 = cols1[::-1]
cols2 = cols2[::-1]
cols3 = cols3[::-1]
cols4 = cols4[::-1]
cols5 = cols5[::-1]
cols6 = cols6[::-1]

a = np.concatenate((cols1[np.newaxis, :], cols3[np.newaxis, :], cols5[np.newaxis, :]), axis=0)
b = np.concatenate((cols2[np.newaxis, :], cols4[np.newaxis, :], cols6[np.newaxis, :]), axis=0)


def Activity_add_noise(x, Inputsize, Inputlength, SNR):
    x_add = x*(np.ones((Inputsize, Inputlength)) + np.random.randn(Inputsize, Inputlength)/SNR)
    return x_add


a5 = Activity_add_noise(a, 3, 302, 5)
a6 = Activity_add_noise(a, 3, 302, 6)
a7 = Activity_add_noise(a, 3, 302, 7)
a8 = Activity_add_noise(a, 3, 302, 8)
a9 = Activity_add_noise(a, 3, 302, 9)
a10 = Activity_add_noise(a, 3, 302, 10)
a11 = Activity_add_noise(a, 3, 302, 11)
a12 = Activity_add_noise(a, 3, 302, 12)
a13 = Activity_add_noise(a, 3, 302, 13)
a14 = Activity_add_noise(a, 3, 302, 14)
a15 = Activity_add_noise(a, 3, 302, 15)
a = np.concatenate((a, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15), axis=0)
b = np.concatenate((b, b, b, b, b, b, b, b, b, b, b, b), axis=0)
x = a[:, :, np.newaxis]
y = b[:, :, np.newaxis]


# Train with new data
model = tf.keras.models.load_model('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\rnn_snr5.h5')
hist = model.fit(x, y, verbose=1, validation_split=0.2, shuffle=True)
print(hist.history)
model.save('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\rnn_PET.h5')
