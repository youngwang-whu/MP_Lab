# -*- coding=utf-8 -*-
# Author: Yiang Wang
# Date: 2018.12.12
# Aim：


from __future__ import absolute_import, division, print_function
import numpy as np
import xlrd


# Read xls files for E=110~118,170~178
ExcelFile110 = xlrd.open_workbook('110ad.xlsx')
ExcelFile112 = xlrd.open_workbook('112ad.xlsx')
ExcelFile114 = xlrd.open_workbook('114ad.xlsx')
ExcelFile116 = xlrd.open_workbook('116ad.xlsx')
ExcelFile118 = xlrd.open_workbook('118ad.xlsx')
ExcelFile170 = xlrd.open_workbook('170ad.xlsx')
ExcelFile172 = xlrd.open_workbook('172ad.xlsx')
ExcelFile174 = xlrd.open_workbook('174ad.xlsx')
ExcelFile176 = xlrd.open_workbook('176ad.xlsx')
ExcelFile178 = xlrd.open_workbook('178ad.xlsx')

ExcelFile = [ExcelFile110, ExcelFile112, ExcelFile114, ExcelFile116, ExcelFile118, ExcelFile170, ExcelFile172,
             ExcelFile174, ExcelFile176, ExcelFile178]


# Load data for E=110~118,170~178
def load_data(ExcelFile):
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

# Divide data into activity and dose
def get_TrainSet(ExcelFile):
    x_train = np.zeros((100, 302))
    y_train = np.zeros((100, 302))
    for i in np.arange(10):
        Activity, Dose = load_data(ExcelFile[i])
        x_train[10*(i):10*(i+1), :] = Activity
        y_train[10*(i):10*(i+1), :] = Dose

    return x_train, y_train
x, y = get_TrainSet(ExcelFile)


# Read xls files for E=140~148MeV
ExcelFile140 = xlrd.open_workbook('140.xlsx')
ExcelFile142 = xlrd.open_workbook('142.xlsx')
ExcelFile144 = xlrd.open_workbook('144.xlsx')
ExcelFile146 = xlrd.open_workbook('146.xlsx')
ExcelFile148 = xlrd.open_workbook('148.xlsx')

# Load data for E=140~148MeV
def LoadData(ExcelFile):
    # Get the contents of sheet
    sheet = ExcelFile.sheet_by_name('Sheet1')
    # Get the whole value of sheet
    cols = []
    for i in np.arange(sheet.ncols):
        cols.append(sheet.col_values(i)[1:])

    return np.array(cols[0:20]), np.array(cols[20:])

activity_140, dose_140 = LoadData(ExcelFile140)
activity_142, dose_142 = LoadData(ExcelFile142)
activity_144, dose_144 = LoadData(ExcelFile144)
activity_146, dose_146 = LoadData(ExcelFile146)
activity_148, dose_148 = LoadData(ExcelFile148)


X = np.concatenate((x, activity_140, activity_142, activity_144, activity_146, activity_148), axis=0)
Y = np.concatenate((y, dose_140, dose_140, dose_142, dose_142, dose_144,
                    dose_144, dose_146, dose_146, dose_148, dose_148), axis=0)

# Add noise
X = X*(np.ones((200, 302)) + np.random.randn(200, 302)*0.15)

# Save training set
np.save('x.npy', X)
np.save('y.npy', Y)
