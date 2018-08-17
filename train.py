from math import ceil
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, LSTM, Dense, Reshape, Input
from keras.callbacks import TensorBoard

batch_size = 512
current_batch = 0
file_name = 'data/training_data.hdf5'
state_name = 'state'
action_name = 'action'


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
    model.add(Dense(3))
    model.compile('adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def read_batch():
    states = []
    actions = []
    try:
        f = h5py.File(file_name, 'a')
        assert state_name in f and action_name in f
        states = f[state_name][current_batch * batch_size:
                               (current_batch + 1) * batch_size]
        actions = f[action_name][current_batch * batch_size:
                                 (current_batch + 1) * batch_size]
        assert len(states) == len(actions)
        print('Read batch of size: {} from {}'.format(batch_size, file_name))
    except Exception as e:
        print(e)
    finally:
        f.close()
    return states, actions


states = []
actions = []
try:
    f = h5py.File(file_name, 'a')
    assert state_name in f and action_name in f
    states = np.reshape(f[state_name][:8192], newshape=(-1, 60, 80, 1))
    actions = f[action_name][:8192]
    assert len(states) == len(actions)
except Exception as e:
    print(e)
finally:
    f.close()

print(np.shape(states))

split = ceil(len(states) * .9)

model = get_model()
model.summary()

tensorboard = TensorBoard(log_dir='./logs', write_graph=True)

model.fit(states[:split], actions[:split], batch_size=batch_size,
          epochs=6, callbacks=[tensorboard])
model.evaluate(states[split:], actions[split:], batch_size=batch_size)
