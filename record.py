import numpy as np
import time
import cv2
from collections import deque
import win32api
import h5py
import threading
from utils.directkeys import get_pressed_keys, keys_to_tract
from utils.get_screen import get_screen
from utils.fps_sync import FpsSync

is_running = True
fps_limit = 16
# 27 refers to ESC key
pause_key = 27
file_name = 'data/training_data.hdf5'
state_name = 'state'
action_name = 'action'
batch_size = 512
# openCV uses shape: (height, width)
state_dim = (60, 80)
action_dim = (len(keys_to_tract),)


def input_to_one_hot(input):
    one_hot = np.zeros(len(keys_to_tract))

    for key in input:
        if key in keys_to_tract:
            one_hot[keys_to_tract.index(key)] = 1

    return one_hot


def save_batch(state, action):
    assert len(state) == len(action) == batch_size

    try:
        f = h5py.File(file_name, 'a')

        if state_name not in f:
            f.create_dataset(
                state_name,
                (batch_size, state_dim[0], state_dim[1]),
                maxshape=(None, state_dim[0], state_dim[1]),
                chunks=(batch_size, state_dim[0], state_dim[1]),
                dtype=np.uint8, data=state),
        else:
            f[state_name].resize(f[state_name].shape[0] + batch_size, axis=0)
            f[state_name][-batch_size:] = state

        if action_name not in f:
            f.create_dataset(
                action_name,
                (batch_size, action_dim[0]),
                maxshape=(None, action_dim[0]),
                chunks=(batch_size, action_dim[0]),
                dtype=np.uint8, data=action),
        else:
            f[action_name].resize(f[action_name].shape[0] + batch_size, axis=0)
            f[action_name][-batch_size:] = action

        assert f[state_name].shape[0] == f[action_name].shape[0]
        print('Saved batch of size: {} in {}\nCurrent data length: {}'
              .format(batch_size, file_name, f[state_name].shape[0]))

    except Exception as e:
        print(e)

    finally:
        f.close()


def main():
    # reset pause key listener
    win32api.GetAsyncKeyState(pause_key)
    iterations = 0
    state_buffer = []
    action_buffer = []
    fps_sync = FpsSync(fps_limit)

    for i in range(3)[::-1]:
        print('starting in {} seconds'.format(i), end='\r')
        time.sleep(1)
    print('\nRecording!')

    fps_sync.init()

    while(is_running):
        if win32api.GetAsyncKeyState(pause_key):
            print('\nstopped')
            break

        # screen = get_screen('Grand Theft Auto V')
        screen = get_screen()
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, state_dim[::-1])
        # Conv layers require shape (x, y, color_space)
        screen = np.reshape(screen, (state_dim[0], state_dim[1], 1))
        pressed_keys = input_to_one_hot(get_pressed_keys())

        thread = None

        state_buffer.append(screen)
        action_buffer.append(pressed_keys)

        if len(state_buffer) >= batch_size or len(action_buffer) >= batch_size:
            if thread and thread.is_alive():
                thread.join()

            thread = threading.Thread(target=save_batch,
                                      args=(state_buffer, action_buffer))
            thread.start()
            state_buffer = []
            action_buffer = []

        iterations += 1

        if iterations % 10 == 0:
            print('fps: {}'.format(fps_sync.get_fps()), end='\r')

        fps_sync.sync()


main()
