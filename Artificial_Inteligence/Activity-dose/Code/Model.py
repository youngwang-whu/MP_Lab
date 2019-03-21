# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Date: 08.19.2018
# Aim：Train, Test and save model
from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def NN_build_model():
    # Build RNN
    model = tf.keras.Sequential()
    # build
    model.add(tf.keras.layers.Dense(
        input_shape=(302, 1),
        units=200, activation=tf.nn.relu
    ))
    model.add(tf.keras.layers.Dense(
        units=15, activation=tf.nn.relu
    ))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    print(model.summary())
    print('training----------------------------------------')

    return model


# 需要调整的参数：units, optimizer, loss
def LSTM_build_model(units):
    # Build RNN
    model = tf.keras.Sequential()
    # build a LSTM layer
    model.add(tf.keras.layers.LSTM(
        input_shape=(302, 1),
        units=units,
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

    return model


def seq2seq_build_model(hidden_dim, num_encoder_tokens, num_decoder_tokens):

    encoder_inputs = tf.keras.layers.Input(shape=(None, num_encoder_tokens))
    encoder = tf.keras.layers.LSTM(hidden_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = tf.keras.layers.Input(shape=(None, num_decoder_tokens))
    decoder_lstm = tf.keras.layers.LSTM(hidden_dim, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_time = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
    decoder_outputs = decoder_time(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = tf.keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    adam = tf.keras.optimizers.Adam(0.006)
    model.compile(optimizer=adam,
              loss='mse')
    print(model.summary())
    print('training---------------------------------')

    return model


def train_model_NN(x_train, y_train, x_test, y_test, batch_size, epochs):
    model = NN_build_model()

    hist = model.fit(x_train, y_train, batch_size, epochs, verbose=1, validation_split=0.2, shuffle=True)
    print(hist.history)

    score = model.evaluate(x_test, y_test, verbose=0)
    print(model.metrics_names)
    print('Test loss:', score)

    return model, hist, score


def train_model_lstm(x_train, y_train, x_test, y_test, units, batch_size, epochs):
    model = LSTM_build_model(units)

    hist = model.fit(x_train, y_train, batch_size, epochs, verbose=1, validation_split=0.2, shuffle=True)
    print(hist.history)

    score = model.evaluate(x_test, y_test, verbose=0)
    print(model.metrics_names)
    print('Test loss:', score)

    return model, hist, score


def train_model_seq2seq(x_train, y_train, x_test, y_test, units, input_length, output_length):
    model = seq2seq_build_model(units, input_length, output_length)

    hist = model.fit(x_train, y_train, batch_size=80, epochs=200, verbose=1, validation_split=0.2, shuffle=True)
    print(hist.history)

    score = model.evaluate(x_test, y_test, verbose=0)
    print(model.metrics_names)
    print('Test loss:', score)

    return model, hist, score




def Get_ModelAccuracy_figure(hist):
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


def Get_ModelLoss_figure(hist):
    #Summarize history for loss
    plt.figure('Loss')
    plt.plot(hist.history['loss'], color='#0000FF', marker='.')
    plt.plot(hist.history['val_loss'], color='#A52A2A', marker='*')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()




def model_predict(x, y, model, dimension):
    figure, ax = plt.subplots(dimension, 1)
    y_predict = model.predict(x, dimension)

    for i in np.arange(dimension):
        ax[i].plot(x[i], y[i], 'b--', x[i], y_predict[i], 'r--')
    plt.show()

    return y_predict
