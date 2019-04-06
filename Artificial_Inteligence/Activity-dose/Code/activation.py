# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Date:
# Aim：
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import Adam


# load data and model
X = np.load("MP_Lab\\Artificial_Inteligence\\Activity-dose\\data\\X.npy")
Y = np.load("MP_Lab\\Artificial_Inteligence\\Activity-dose\\data\\Y.npy")
model = load_model("MP_Lab\\Artificial_Inteligence\\Activity-dose\\data\\rnn_model_0903.h5")


# 以这个model的预测值作为输出
extract_state_model = Model(
    inputs=model.input,
    outputs=model.get_layer('lstm_1').output
)
h_state = extract_state_model.predict(X[:, :, np.newaxis])


y_predict = model.predict(X[:, ::-1, np.newaxis])

x = np.concatenate((X[:, 50:100], X[:, 0:50], X[:, 100:302]), axis=1)
y = np.concatenate((Y[:, 50:100], Y[:, 0:50], Y[:, 100:302]), axis=1)
y_predict = model.predict(x[:, :, np.newaxis])


# Input a 1D array.
def normalize(data):
    mx = max(data)
    mn = min(data)
    return [(float(i) - mn) / (mx-mn) for i in data]

# 反向
c = np.arange(0, 302)
plt.plot(c, normalize(y_predict[1, :]), 'r')
plt.plot(c, X[1, ::-1], 'b')
plt.plot(c, Y[1, ::-1], 'g')
plt.show()
# 切片调整
c = np.arange(0, 302)
plt.plot(c, normalize(y_predict[1, :]), 'r')
plt.plot(c, x[1, :], 'b')
plt.plot(c, y[1, :], 'g')
plt.show()

# plot activation
fig, ax = plt.subplots(figsize=(40, 20))
b = np.swapaxes(h_state[14], 0, 1)
c = ax.pcolor(b, cmap='RdBu', vmin=-1, vmax=1)
ax.set_xticklabels([0, 50, 100, 150, 200, 250, 300], fontsize=40) # 坐标轴的刻度
ax.set_yticklabels([0, 5, 10, 15, 20, 25, 30], fontsize=40)
ax.set_xlabel('Depth', fontsize=50)
ax.set_ylabel('Units', fontsize=50)
#ax0.set_title('default: no edges')
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
e = fig.colorbar(c, ax=ax)
e.ax.tick_params(labelsize='large')  # 调整 coloebar 字体的
# 带网格线的
# c = ax[1].pcolor(b, edgecolors='k', linewidths=1)
# ax1.set_title('thick edges')
#fig.tight_layout()
plt.show()

