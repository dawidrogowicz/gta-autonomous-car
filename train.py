from keras.models import Sequential
from keras.layers import Conv2D, LSTM, Dense, Flatten

batch_size = 200
current_batch = 0
file_name = 'data/training_data.hdf5'
state_name = 'state'
action_name = 'action'


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

    def get_model():
        model = Sequential()
        model.add(Conv2D(24, (5, 5), strides=(3, 3)))  # size: (26, 20, 24)
        model.add(Conv2D(36, (5, 5), strides=(2, 2)))  # size: (12, 9, 36)
        model.add(Conv2D(48, (5, 5), strides=(2, 2)))  # size: (5, 3, 48)
        model.add(Conv2D(64, (3, 3), strides=(2, 2)))  # size: (2, 1, 64)
        model.add(Conv2D(64, (2, 1)))  # size: (1, 1, 64)
        model.add(Flatten())  # size: (64,)
        model.add(LSTM(64))
        model.add(Dense(512))
        model.add(Dense(128))
        model.add(Dense(64))
        model.add(Dense(3))
        model.compile('adam', loss='mean_squared_error', metrics=['accuracy'])
        return model
