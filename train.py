from math import ceil
from subprocess import call
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, LSTM, Dense, Reshape, Input
from keras.callbacks import TensorBoard

batch_size = 256
train_split = .8
epochs = 4
file_name = 'data/training_data.hdf5'
state_name = 'state'
action_name = 'action'


def get_data_len(train=False, test=False):
    data_len = 0
    try:
        f = h5py.File(file_name, 'a')
        assert state_name in f and action_name in f
        assert len(f[state_name]) == len(f[action_name])
        data_len = len(f[state_name])
    except Exception as e:
        print(e)
    finally:
        f.close()
    if train:
        data_len *= train_split
    elif test:
        data_len *= 1 - train_split
    return data_len


def get_model():
    model = Sequential()
    # size: (19, 26, 24)
    model.add(Conv2D(24, 5, strides=3, input_shape=(60, 80, 1)))
    model.add(Conv2D(36, 5, strides=2))  # size: (8, 11, 36)
    model.add(Conv2D(48, 3, strides=2))  # size: (3, 5, 48)
    model.add(Conv2D(64, 3, strides=2))  # size: (1, 2, 64)
    model.add(Conv2D(64, (1, 2)))  # size: (1, 1, 64)
    model.add(Reshape((64, 1)))  # size: (64,)
    model.add(LSTM(64))
    model.add(Dense(512))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))
    model.compile('adam', loss='mean_squared_error', metrics=['mae'])
    return model


def read_batch(file, test=False):
    assert state_name in file and action_name in file
    assert len(file[state_name]) == len(file[action_name])
    data_size = len(file[state_name])
    start = int(data_size * train_split) if test else 0
    end = data_size if test else int(data_size * train_split)
    step = batch_size
    while True:
        for i in range(start, end, step):
            batch_end = i + step
            if batch_end > data_size:
                batch_end = data_size
            states = file[state_name][i:batch_end]
            actions = file[action_name][i:batch_end]
            yield states, actions


model = get_model()
model.summary()

tensorboard = TensorBoard(log_dir='./log', write_graph=False, write_images=True)

print('Training...')
try:
    file = h5py.File(file_name, 'a')
    model.fit_generator(read_batch(file), epochs=epochs, steps_per_epoch=ceil(
        get_data_len(train=True) / batch_size), callbacks=[tensorboard])
except Exception as e:
    print(e)
finally:
    file.close()

test_states = []
test_actions = []
try:
    f = h5py.File(file_name, 'a')
    assert state_name in f and action_name in f
    assert len(f[state_name]) == len(f[action_name])
    data_len = len(f[state_name])
    split = int(train_split * data_len)
    test_states = f[state_name][split:]
    test_actions = f[action_name][split:]
except Exception as e:
    print(e)
finally:
    f.close()

score, mae = model.evaluate(test_states, test_actions, batch_size=batch_size)
print('Score: {}\nMAE: {}'.format(score, mae))

call(['tensorboard', '--logdir', './log/'])
