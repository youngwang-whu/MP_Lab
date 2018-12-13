# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Date: 07.13.2018
# Aim：

from __future__ import absolute_import, division, print_function
import numpy as np
import xlrd
import tensorflow as tf
import skimage
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier


# Load the xls file
xlsx136 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\class_energy\\136.xlsx')
xlsx138 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\class_energy\\138.xlsx')
xlsx140 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\class_energy\\140.xlsx')
xlsx142 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\class_energy\\142.xlsx')
xlsx144 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\class_energy\\144.xlsx')


# Get the contents of sheet
sheet136 = xlsx136.sheet_by_name('Sheet1')
sheet138 = xlsx138.sheet_by_name('Sheet1')
sheet140 = xlsx140.sheet_by_name('Sheet1')
sheet142 = xlsx142.sheet_by_name('Sheet1')
sheet144 = xlsx144.sheet_by_name('Sheet1')
print(sheet136.name, sheet136.nrows, sheet136.ncols)


# Get the whole value of sheet(excel中一定不能有占位符！！！不然会被认为都是字符格式的)
x_train = []
y_train = []
for i in range(0, 100):
    col_136 = sheet136.col_values(i)
    x_train.append(col_136)
    y_train.append(0)

    col_138 = sheet138.col_values(i)
    x_train.append(col_138)
    y_train.append(1)

    col_140 = sheet140.col_values(i)
    x_train.append(col_140)
    y_train.append(2)

    col_142 = sheet142.col_values(i)
    x_train.append(col_142)
    y_train.append(3)

    col_144 = sheet144.col_values(i)
    x_train.append(col_144)
    y_train.append(4)


# Translate list into array
x_train = np.array(x_train)
y_train = np.array(y_train)

# 先保存一部分值作为测试集
x_test = x_train[0:99:2, :]
y_test = y_train[0:99:2]

# 引入部分随机噪声
x_train[0:99:2, :] = skimage.util.random_noise(x_train[0:99:2], mode='gaussian', seed=None, clip=True)

# Converts a class vector (integers) to binary class matrix.
# 并不清楚这个函数的num_classes这个参数是否需要？？？
y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=5)


# Build NN
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(units=1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=126, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=5, activation=tf.nn.softmax))

# Define optimizer
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['categorical_accuracy'])

# Training
print('Training')
hist = model.fit(x_train, y_train, batch_size=64, epochs=1000, validation_split=0.2, shuffle=True, verbose=1)
print(hist.history)

# testing
score = model.evaluate(x_test, y_test, verbose=0)
print(model.metrics_names)
print('Test loss:', score)


# List all data in history
print(hist.history.keys())
# Summarize history for accuracy
plt.figure(1)
plt.plot(hist.history['categorical_accuracy'], color='#0000FF', marker='.')
plt.plot(hist.history['val_categorical_accuracy'], color='#A52A2A', marker='*')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.figure(2)
plt.plot(hist.history['loss'], color='#0000FF', marker='.')
plt.plot(hist.history['val_loss'], color='#A52A2A', marker='*')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Predicting & Confusion matrix
pred_y = model.predict(x_test)
pred_label = np.argmax(pred_y, axis=1)
true_label = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(true_label, pred_label)
print(conf_matrix)

"""
# Predicting
a = x_test[24, :]
predict_labels = model.predict(a)
print("number:", a, "\n", "prediction", predict_labels)

# 输出预测结果
predict = model.predict_classes(x_test)
# 计算预测精度
p = tf.keras.utils.to_categorical(predict, num_classes=5)
a = tf.keras.metrics.categorical_accuracy(y_test, predict)
"""


