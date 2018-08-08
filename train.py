import numpy as np
import time
import cv2
import h5py
from directkeys import get_pressed_keys, keys
from get_screen import get_screen

is_running = True
file_name = 'data/training_data.hdf5'
state_dset_name = 'state'
action_dset_name = 'action'
batch_size = 200
# openCV uses shape: (height, width)
state_dim = (60, 80)
action_dim = (len(keys),)


def input_to_one_hot(input):
    one_hot = np.zeros(len(keys))

    for key in input:
        if key in keys:
            one_hot[list(keys.keys()).index(key)] = 1

    return one_hot


def save_batch(state, action):
    assert len(state) == len(action) == batch_size

    try:
        f = h5py.File(file_name, 'a')

        if state_dset_name not in f:
            f.create_dataset(state_dset_name,
                             (batch_size, state_dim[0], state_dim[1]),
                             maxshape=(None, state_dim[0], state_dim[1]),
                             chunks=(batch_size, state_dim[0], state_dim[1]),
                             dtype=np.uint8),
        if action_dset_name not in f:
            f.create_dataset(action_dset_name,
                             (batch_size, action_dim[0]),
                             maxshape=(None, action_dim[0]),
                             chunks=(batch_size, action_dim[0]),
                             dtype=np.uint8),

        state_dset = f[state_dset_name]
        action_dset = f[action_dset_name]

        assert state_dset.shape[0] == action_dset.shape[0]

        state_dset.resize(state_dset.shape[0] + batch_size, axis=0)
        action_dset.resize(action_dset.shape[0] + batch_size, axis=0)
        state_dset[-batch_size:] = state
        action_dset[-batch_size:] = action
        print(3333, state_dset.shape)
        print(4444, action_dset.shape)
    except Exception as e:
        print(e)
    finally:
        f.close()
        print('file closed')


def main():
    start = time.time()
    iterations = 0
    state_buffer = []
    action_buffer = []

    while(is_running):
        screen = get_screen('Grand Theft Auto V')
        # screen = get_screen()
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, state_dim[::-1])
        pressed_keys = input_to_one_hot(get_pressed_keys())

        if (cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()

        state_buffer.append(screen)
        action_buffer.append(pressed_keys)

        if len(state_buffer) >= batch_size or len(action_buffer) >= batch_size:
            save_batch(state_buffer, action_buffer)
            print('Saved chunk of size: {}\nfps: {}'.format(len(state_buffer),
                                                            iterations / total))
            state_buffer = []
            action_buffer = []

        iterations += 1
        total = time.time() - start


main()
