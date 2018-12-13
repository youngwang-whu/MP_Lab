# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Date: 2018.12.12
# Aim：
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
x_p = a[:, :, np.newaxis]
y_p = b[:, :, np.newaxis]


# Load model
model = tf.keras.models.load_model('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\rnn_model.h5')
# Predict
y_pred = model.predict(x_p)
# Plot figure of 3 situations
c = np.arange(0, 302)
for i in np.arange(3):
    plt.figure(i)
    plt.plot(c, y_pred[i, :], color='#0000FF', linestyle='--')
    plt.plot(c, y_p[i, :], color='#8B0000', linestyle='--')
    plt.plot(c, x_p[i,:], color='#FFD700', linestyle='--')
    plt.show()
