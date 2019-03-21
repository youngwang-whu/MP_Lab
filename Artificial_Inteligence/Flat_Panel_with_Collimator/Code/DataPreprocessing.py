# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:34:35 2019

@author: Yiang Wang
"""

# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Date: 02/18/2019
# Aim：
from __future__ import absolute_import, division, print_function

import numpy as np
import xlrd


# Load validated data
ExcelFile = xlrd.open_workbook('collimator_H3.xlsx')
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
    P = np.empty((8, 150))
    for i in range(8):
        P[i, :] = normalize(np.array(Pi.col_values(i+1)[1:151]))
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
np.save("MP_Lab\\Artificial_Inteligence\\Flat_Panel_with_Collimator\\data\\collimator_H3_X.npy", X)
np.save("MP_Lab\\Artificial_Inteligence\\Flat_Panel_with_Collimator\\data\\collimator_H3_Y.npy", Y)

