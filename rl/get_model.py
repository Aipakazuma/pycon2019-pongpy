import tensorflow as tf


def v1(input_shape, n_output):
    m = tf.keras.models.Sequential()
    m.add(tf.keras.layers.Dense(64, activation='relu',
                                input_shape=input_shape))
    m.add(tf.keras.layers.Dense(256, activation='relu'))
    m.add(tf.keras.layers.Dense(n_output,
                                activation='softmax'))

    return m
