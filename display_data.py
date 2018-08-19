import numpy as np
import time
import cv2
from collections import deque
from utils.directkeys import keys_to_tract
import win32api
import h5py

is_running = True
fps_limit = 30
fps_check_interval = 1
fps_limit_treshold = 1 / fps_limit * 0.2
# 27 refers to ESC key
pause_key = 27
file_name = 'data/training_data.hdf5'
state_name = 'state'
action_name = 'action'
preview_size = (800, 600)


def from_one_hot(one_hot):
    return [keys_to_tract[i] for i, val in enumerate(one_hot) if val == 1]


def main():
    # reset pause key listener
    win32api.GetAsyncKeyState(pause_key)
    iterations = 0
    state_dset = []
    action_dset = []
    frametimes = deque(maxlen=fps_limit * fps_check_interval)
    frametimes.append(time.time())

    try:
        f = h5py.File(file_name, 'a')

        assert state_name in f and action_name in f

        state_dset = f[state_name][:]
        action_dset = f[action_name][:]

    except Exception as e:
        print(e)

    finally:
        f.close()

    for state, action in zip(state_dset, action_dset):
        if win32api.GetAsyncKeyState(pause_key):
            print('\nstopped')
            cv2.destroyAllWindows()
            break

        state = cv2.resize(state, preview_size)
        pressed_keys = from_one_hot(action)

        cv2.imshow('preview', state)
        if (cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()

        print(pressed_keys)

        iterations += 1

        if ((1 / fps_limit) >
                (time.time() - frametimes[-1]) + fps_limit_treshold):
            time.sleep((1 / fps_limit) - (time.time() - frametimes[-1]))

        frametimes.append(time.time())


main()
