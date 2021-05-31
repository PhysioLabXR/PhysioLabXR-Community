import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.models import load_model
from IPython.display import clear_output
from tensorflow.python.keras import Sequential, Model, Input
from tensorflow.python.keras.layers import TimeDistributed, Conv2D, BatchNormalization, MaxPooling2D, Flatten, \
    concatenate, LSTM, Dropout, Dense

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib


def make_model_dl_final():
    encoder1 = Sequential()
    encoder1.add(tf.keras.layers.InputLayer(input_shape=(120, 8, 16, 1)))
    encoder1.add(TimeDistributed(Conv2D(filters=8, kernel_size=(2, 3),
                                        data_format='channels_last')))
    encoder1.add(TimeDistributed(tf.keras.layers.LeakyReLU(alpha=0.1)))

    encoder1.add(TimeDistributed(MaxPooling2D(pool_size=2)))
    encoder1.add(TimeDistributed(BatchNormalization()))
    encoder1.add(TimeDistributed(Flatten()))  # or Flatten()
    encoder1.add(TimeDistributed(Dense(32, activation='relu')))
    # encoder1.add(TimeDistributed(Dropout(rate=0.2)))

    # ENcoder2
    encoder2 = Sequential()
    encoder2.add(tf.keras.layers.InputLayer(input_shape=(120, 8, 64, 1)))
    encoder2.add(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3),
                                        data_format='channels_last')))
    encoder2.add(TimeDistributed(tf.keras.layers.LeakyReLU(alpha=0.1)))
    encoder2.add(TimeDistributed(MaxPooling2D(pool_size=2)))
    encoder2.add(TimeDistributed(BatchNormalization()))
    encoder2.add(TimeDistributed(Flatten()))  # or Flatten()
    encoder2.add(TimeDistributed(Dense(64, activation='relu')))
    # encoder2.add(TimeDistributed(Dropout(rate=0.2)))

    merged = concatenate([encoder1.output, encoder2.output])
    # merged_out = LSTM(32, return_sequences=True, kernel_initializer='random_uniform',
    #                   kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
    #                   recurrent_regularizer=tf.keras.regularizers.l2(l=1e-5),
    #                   activity_regularizer=tf.keras.regularizers.l2(l=1e-5)
    #                   )(merged)
    # merged_out = Dropout(rate=0.2)(merged_out)
    merged_out = LSTM(32,
                      kernel_initializer='random_uniform',
                      return_sequences=False,
                      )(merged)
    merged_out = Dropout(rate=0.2)(merged_out)
    merged_out = Dense(256,
                       activation='relu'
                       )(merged_out)
    merged_out = Dropout(rate=0.2)(merged_out)
    merged_out = Dense(30, activation='softmax', kernel_initializer='random_uniform')(merged_out)
    # encoder1.build((None, 120, 8, 16, 1))
    # encoder1.summary()
    model = Model(inputs=[encoder1.input, encoder2.input], outputs=merged_out)

    adam = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-7)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model