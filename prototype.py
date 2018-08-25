from math import ceil
from time import time
from subprocess import call
from keras.models import Sequential, load_model
from keras.utils.io_utils import HDF5Matrix
from keras.layers import Conv2D, LSTM, Dense, Reshape, Input
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import numpy as np
import h5py
import sys
import os

batch_size = 256
train_split = .8
epochs = 24
log_dir = 'log/prototype/'
data_path = 'data/training_data.hdf5'
model_path_template = 'models/prototype.{id}.hdf5'
state_name = 'state'
action_name = 'action'


def set_epoch_num(filepath, num):
    dset_name = 'model_weights'
    with h5py.File(filepath) as f:
        f[dset_name].attrs['last_epoch'] = num


class LastEpoch(Callback):
    def __init__(self, filepath):
        super(LastEpoch, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, _):
        set_epoch_num(self.filepath, epoch)


def get_last_epoch_num(filepath):
    dset_name = 'model_weights'
    attr = 'last_epoch'
    with h5py.File(filepath) as f:
        return f[dset_name].attrs[attr] if attr in f[dset_name].attrs else 0


def get_sizes():
    len = HDF5Matrix(data_path, state_name).end
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
    x = HDF5Matrix(data_path, state_name)
    y = HDF5Matrix(data_path, action_name)
    assert x.end == y.end
    train_len, len, *_ = get_sizes()
    start = train_len if test else 0
    end = len if test else train_len
    while True:
        for i in range(start, end, batch_size):
            batch_end = len if i + batch_size > len else i + batch_size
            yield x[i:batch_end], y[i:batch_end]


def train(model_id=None):
    run_id = model_id if model_id else '{:x}'.format(int(time()))
    model_path = model_path_template.format(id=run_id)
    init_epoch = 0
    if model_id and os.path.isfile(model_path):
        model = load_model(model_path)
        init_epoch = get_last_epoch_num(model_path) + 1
    else:
        model = get_model()
    last_epoch = LastEpoch(model_path)
    log = TensorBoard(log_dir=os.path.join(log_dir, run_id), write_graph=False)
    checkpoint = ModelCheckpoint(model_path, monitor='val_mean_absolute_error')
    train_len, _, train_batches, batches = get_sizes()
    model.fit_generator(read_batch(), steps_per_epoch=train_batches,
                        validation_data=read_batch(test=True),
                        validation_steps=batches - train_batches,
                        initial_epoch=init_epoch, epochs=epochs,
                        callbacks=[log, checkpoint, last_epoch])
    call(['tensorboard', '--logdir', log_dir])


if __name__ == '__main__':
    if not os.path.isdir('models'):
        os.mkdir('models')
    model_id = sys.argv[1] if len(sys.argv) > 1 else None
    train(model_id)
