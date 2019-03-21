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


# Notice: Modify "\\" into "/" in Linux server!
# X = np.load("MP_Lab/Artificial_Inteligence/Flat_Panel_with_Collimator/data/collimator_H3_X.npy")
# Y = np.load("MP_Lab/Artificial_Inteligence/Flat_Panel_with_Collimator/data/collimator_H3_Y.npy")
X = np.load("MP_Lab\\Artificial_Inteligence\\Flat_Panel_with_Collimator\\data\\collimator_H3_X.npy")
Y = np.load("MP_Lab\\Artificial_Inteligence\\Flat_Panel_with_Collimator\\data\\collimator_H3_Y.npy")
X1 = X[:, 0:150:2]
X2 = X[:, 1:150:2]
X3 = (X1 + X2)/2
x_train = X[:, :, np.newaxis]
y_train = Y[:, :, np.newaxis]

import random
import json
from six.moves import range
import six


def sequence_pad(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    `(num_samples, num_timesteps)`.

    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(num_samples, maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


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


model = load_model("collimator_lstm.h5")

