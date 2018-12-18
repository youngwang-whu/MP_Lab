# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Aim:
# Date:

from __future__ import absolute_import, division, print_function

import xlrd
import pandas
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import f_regression


#
# Read xls files when E=140, 142, 144, 146, 148
ExcelFile140 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\140\\140.xlsx')
ExcelFile142 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\142\\142.xlsx')
ExcelFile144 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\144\\144.xlsx')
ExcelFile146 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\146\\146.xlsx')
ExcelFile148 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\148\\148.xlsx')

def load_data1(ExcelFile):
    # Get the contents of sheet
    sheet = ExcelFile.sheet_by_name('Sheet1')
    # Get the whole value of sheet
    # Highlight:通过列表索引的方法，简单解决了读取excel列表时产生空格的问题
    cols = []
    for i in np.arange(sheet.ncols):
        cols.append(sheet.col_values(i)[1:])

    return np.array(cols[0:20])

activity_140 = load_data1(ExcelFile140)
activity_142 = load_data1(ExcelFile142)
activity_144 = load_data1(ExcelFile144)
activity_146 = load_data1(ExcelFile146)
activity_148 = load_data1(ExcelFile148)

X1 = np.concatenate((activity_140, activity_142, activity_144, activity_146, activity_148), axis=0)


#
# Read xls files when E=110~118, 170~178
ExcelFile110 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\110ad.xlsx')
ExcelFile112 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\112ad.xlsx')
ExcelFile114 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\114ad.xlsx')
ExcelFile116 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\116ad.xlsx')
ExcelFile118 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\118ad.xlsx')
ExcelFile170 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\170ad.xlsx')
ExcelFile172 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\172ad.xlsx')
ExcelFile174 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\174ad.xlsx')
ExcelFile176 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\176ad.xlsx')
ExcelFile178 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DataRegression\\178ad.xlsx')

ExcelFile = [ExcelFile110, ExcelFile112, ExcelFile114, ExcelFile116, ExcelFile118, ExcelFile170, ExcelFile172,
             ExcelFile174, ExcelFile176, ExcelFile178]

def load_data2(ExcelFile):
    # Get the contents of sheet
    sheet = ExcelFile.sheet_by_name('Sheet1')
    # Get the whole value of sheet
    # Highlight:通过列表索引的方法，简单解决了读取excel列表时产生空格的问题
    Activity = []
    Dose = []
    for i in np.arange(sheet.ncols):
        # There must be 2 equal signs in "if statement"!
        if i%2 == 0 :
            Activity.append(sheet.col_values(i)[1:])
        else:
            Dose.append(sheet.col_values(i)[1:])

    return np.array(Activity), np.array(Dose)

def get_TrainSet(ExcelFile):
    x_train = np.zeros((100, 302))
    y_train = np.zeros((100, 302))
    for i in np.arange(10):
        Activity, Dose = load_data2(ExcelFile[i])
        x_train[10*(i):10*(i+1), :] = Activity
        y_train[10*(i):10*(i+1), :] = Dose

    return x_train

X2 = get_TrainSet(ExcelFile)


# Get all data of activity
X = np.vstack((X1, X2))


#
# Read all data of range
excel_range = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\range_data.xlsx')

def get_range_data(ExcelFile):
    sheet = ExcelFile.sheet_by_name('Sheet1')
    FullHeightRange = []
    HalfHeightRange = []
    for i in np.arange(45):
        if i%3 == 1:
            FullHeightRange.append(sheet.row_values(i, 1, 11))
        elif i%3 == 2:
            HalfHeightRange.append(sheet.row_values(i, 1, 11))
        else:
            pass
    return np.array(FullHeightRange).flatten(), np.array(HalfHeightRange).flatten()

FullHeightRange, HalfHeightRange = get_range_data(excel_range)

Y1 = np.concatenate((FullHeightRange[50:60], FullHeightRange[50:60], FullHeightRange[60:70], FullHeightRange[60:70],
                    FullHeightRange[70:80], FullHeightRange[70:80], FullHeightRange[80:90], FullHeightRange[80:90],
                     FullHeightRange[90:100], FullHeightRange[90:100], FullHeightRange[0:50], FullHeightRange[100:150]), axis=0)

Y2 = np.concatenate((HalfHeightRange[50:60], HalfHeightRange[50:60], HalfHeightRange[60:70], HalfHeightRange[60:70],
                    HalfHeightRange[70:80], HalfHeightRange[70:80], HalfHeightRange[80:90], HalfHeightRange[80:90],
                     HalfHeightRange[90:100], HalfHeightRange[90:100], HalfHeightRange[0:50], HalfHeightRange[100:150]), axis=0)

x_train = X[50:, :]
y1_train = Y1[50:]
y1_train = y1_train[:, np.newaxis]
y2_train = Y2[50:]
y2_train = y2_train[:, np.newaxis]

x_test = X[0:50, :]
y1_test = Y1[0:50]
y1_test = y1_test[:, np.newaxis]
y2_test = Y2[0:50]
y2_test = y2_test[:, np.newaxis]


#
# Build RNN
model = tf.keras.Sequential()
# build
model.add(tf.keras.layers.Dense(
        input_shape=(302, ), units=100, activation=tf.nn.relu
    ))
model.add(tf.keras.layers.Dense(
        units=15, activation=tf.nn.relu
    ))
model.add(tf.keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='sgd')
print('training----------------------------------------')



hist = model.fit(x_train, y1_train, batch_size=40, epochs=2000, verbose=1, validation_split=0.2, shuffle=True)
print(hist.history)
print(model.summary())

score = model.evaluate(x_test, y_test, verbose=0)
print(model.metrics_names)
print('Test loss:', score)



model1, hist1, score1 = train_model_NN(x_train, y1_train, x_test, y1_test, batch_size=75, epochs=100)




# #############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
x_select = f_regression(x_train, y1_train)



clf = MLPRegressor(solver='lbfgs', alpha=1e-5, verbose=True, max_iter=1000,
                    hidden_layer_sizes=(200, 20), random_state=1)

predict1 = clf.fit(x_select, y1_train).predict(x_test)
