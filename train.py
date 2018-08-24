from math import ceil
from subprocess import call
import numpy as np
from keras.models import Sequential
from keras.utils.io_utils import HDF5Matrix
from keras.layers import Conv2D, LSTM, Dense, Reshape, Input
from keras.callbacks import TensorBoard

batch_size = 256
train_split = .8
epochs = 2
file_name = 'data/training_data.hdf5'
state_name = 'state'
action_name = 'action'


def get_sizes():
    len = HDF5Matrix(file_name, state_name).end
    batches = ceil(len / batch_size)
    train_batches = ceil(batches * train_split)
    train_len = train_batches * batch_size
    return train_len, len, train_batches, batches


def get_model():
    model = Sequential()
    model.add(Conv2D(24, 5, strides=3, input_shape=(60, 80, 1)))
    model.add(Conv2D(36, 5, strides=2))
    model.add(Conv2D(48, 3, strides=2))
    model.add(Conv2D(64, 3, strides=2))
    model.add(Conv2D(64, (1, 2)))
    model.add(Reshape((64, 1)))
    model.add(LSTM(64))
    model.add(Dense(512))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))
    model.compile('adam', loss='mean_squared_error', metrics=['mae'])
    return model


def read_batch(test=False):
    x = HDF5Matrix(file_name, state_name)
    y = HDF5Matrix(file_name, action_name)
    assert x.end == y.end
    train_len, len, *_ = get_sizes()
    start = train_len if test else 0
    end = len if test else train_len
    while True:
        for i in range(start, end, batch_size):
            batch_end = len if i + batch_size > len else i + batch_size
            yield x[i:batch_end], y[i:batch_end]


model = get_model()
tensorboard = TensorBoard(log_dir='./log', write_graph=False)
train_len, _, train_batches, batches = get_sizes()

print('Training {} batches'.format(train_batches))
model.fit_generator(read_batch(), epochs=epochs, steps_per_epoch=train_batches,
                    callbacks=[tensorboard])
print('Evaluating {} batches'.format(batches - train_batches))
score, mae = model.evaluate_generator(read_batch(test=True),
                                      steps=batches - train_batches)
print('Score: {}\nMAE: {}'.format(score, mae))

call(['tensorboard', '--logdir', './log/'])
