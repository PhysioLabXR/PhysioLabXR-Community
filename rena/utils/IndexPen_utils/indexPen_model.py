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

from config_signal import rd_shape, ra_shape


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
    merged_out = Dense(5, activation='softmax', kernel_initializer='random_uniform')(merged_out)
    # encoder1.build((None, 120, 8, 16, 1))
    # encoder1.summary()
    model = Model(inputs=[encoder1.input, encoder2.input], outputs=merged_out)

    adam = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-7)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def make_model_leo(classes, points_per_sample, channel_mode='channels_last', batch_size=32):
    # creates the Time Distributed CNN for range Doppler heatmap ##########################
    mmw_rdpl_input = (int(points_per_sample),) + rd_shape + (1,) if channel_mode == 'channels_last' else (points_per_sample, 1) + rd_shape
    mmw_rdpl_TDCNN = Sequential()
    mmw_rdpl_TDCNN.add(
        TimeDistributed(
            Conv2D(filters=8, kernel_size=(3, 3), data_format=channel_mode,
                   kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                   bias_regularizer=tf.keras.regularizers.l2(l=1e-5),
                   activity_regularizer=tf.keras.regularizers.l2(l=1e-5),
                   kernel_initializer='random_uniform'),
            input_shape=mmw_rdpl_input))  # use batch input size to avoid memory error
    mmw_rdpl_TDCNN.add(TimeDistributed(tf.keras.layers.LeakyReLU(alpha=0.1)))
    mmw_rdpl_TDCNN.add(TimeDistributed(BatchNormalization()))
    mmw_rdpl_TDCNN.add(TimeDistributed(
        Conv2D(filters=16, kernel_size=(3, 3),
               kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
               bias_regularizer=tf.keras.regularizers.l2(l=1e-5),
               activity_regularizer=tf.keras.regularizers.l2(l=1e-5)
               )))
    mmw_rdpl_TDCNN.add(TimeDistributed(tf.keras.layers.LeakyReLU(alpha=0.1)))
    mmw_rdpl_TDCNN.add(TimeDistributed(BatchNormalization()))
    mmw_rdpl_TDCNN.add(TimeDistributed(MaxPooling2D(pool_size=2)))
    # mmw_rdpl_TDCNN.add(TimeDistributed(
    #     Conv2D(filters=32, kernel_size=(3, 3),
    #            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
    #            bias_regularizer=tf.keras.regularizers.l2(l=0.01))))
    # mmw_rdpl_TDCNN.add(TimeDistributed(tf.keras.layers.LeakyReLU(alpha=0.1)))
    # mmw_rdpl_TDCNN.add(TimeDistributed(BatchNormalization()))
    # mmw_rdpl_TDCNN.add(TimeDistributed(MaxPooling2D(pool_size=2)))
    mmw_rdpl_TDCNN.add(TimeDistributed(Flatten()))  # this should be where layers meets

    # creates the Time Distributed CNN for range Azimuth heatmap ###########################
    mmw_razi_input = (int(points_per_sample),) + ra_shape + (1,) if channel_mode == 'channels_last' else (points_per_sample, 1) + ra_shape
    mmw_razi_TDCNN = Sequential()
    mmw_razi_TDCNN.add(
        TimeDistributed(
            Conv2D(filters=8, kernel_size=(3, 3), data_format=channel_mode,
                   kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                   bias_regularizer=tf.keras.regularizers.l2(l=1e-5),
                   activity_regularizer=tf.keras.regularizers.l2(l=1e-5),
                   kernel_initializer='random_uniform'),
            input_shape=mmw_razi_input))  # use batch input size to avoid memory error
    mmw_razi_TDCNN.add(TimeDistributed(tf.keras.layers.LeakyReLU(alpha=0.1)))
    mmw_razi_TDCNN.add(TimeDistributed(BatchNormalization()))
    mmw_razi_TDCNN.add(TimeDistributed(
        Conv2D(filters=16, kernel_size=(3, 3),
               kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
               bias_regularizer=tf.keras.regularizers.l2(l=1e-5),
               activity_regularizer=tf.keras.regularizers.l2(l=1e-5)
               )))
    mmw_razi_TDCNN.add(TimeDistributed(tf.keras.layers.LeakyReLU(alpha=0.1)))
    mmw_razi_TDCNN.add(TimeDistributed(BatchNormalization()))
    mmw_razi_TDCNN.add(TimeDistributed(MaxPooling2D(pool_size=2)))
    # mmw_razi_TDCNN.add(TimeDistributed(
    #     Conv2D(filters=32, kernel_size=(3, 3), data_format=channel_mode,
    #            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
    #            bias_regularizer=tf.keras.regularizers.l2(l=0.01))))
    # mmw_rdpl_TDCNN.add(TimeDistributed(tf.keras.layers.LeakyReLU(alpha=0.1)))
    # mmw_razi_TDCNN.add(TimeDistributed(BatchNormalization()))
    # mmw_razi_TDCNN.add(TimeDistributed(MaxPooling2D(pool_size=2)))
    mmw_razi_TDCNN.add(TimeDistributed(Flatten()))  # this should be where layers meets

    merged = concatenate([mmw_rdpl_TDCNN.output, mmw_razi_TDCNN.output])  # concatenate two feature extractors
    regressive_tensor = LSTM(units=32, return_sequences=True, kernel_initializer='random_uniform',
                             kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                             recurrent_regularizer=tf.keras.regularizers.l2(l=1e-5),
                             activity_regularizer=tf.keras.regularizers.l2(l=1e-5)
                             )(merged)
    regressive_tensor = Dropout(rate=0.5)(regressive_tensor)
    regressive_tensor = LSTM(units=32, return_sequences=False, kernel_initializer='random_uniform',
                             kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                             recurrent_regularizer=tf.keras.regularizers.l2(l=1e-5),
                             activity_regularizer=tf.keras.regularizers.l2(l=1e-5)
                             )(regressive_tensor)
    regressive_tensor = Dropout(rate=0.5)(regressive_tensor)

    regressive_tensor = Dense(units=256,
                              kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                              bias_regularizer=tf.keras.regularizers.l2(l=1e-5),
                              activity_regularizer=tf.keras.regularizers.l2(l=1e-5)
                              )(regressive_tensor)
    regressive_tensor = Dropout(rate=0.5)(regressive_tensor)
    regressive_tensor = Dense(len(classes), activation='softmax', kernel_initializer='random_uniform')(regressive_tensor)

    model = Model(inputs=[mmw_rdpl_TDCNN.input, mmw_razi_TDCNN.input], outputs=regressive_tensor)
    adam = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-7)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model