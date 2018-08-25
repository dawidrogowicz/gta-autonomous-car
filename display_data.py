import numpy as np
import time
import cv2
from collections import deque
import win32api
import h5py

is_running = True
fps_limit = 16
fps_check_interval = 1
fps_limit_treshold = 1 / fps_limit * 0.2
# 27 refers to ESC key
pause_key = 27
file_name = 'data/training_data.hdf5'
state_name = 'state'
action_name = 'action'
preview_size = (800, 600)


def draw_angle(img, action):
    height = int(preview_size[1] * .9)
    return cv2.line(img, (int(preview_size[0] / 2), height),
                    (int(preview_size[0] * (action + 1) / 2),
                     height), (255, 255, 255), 16)


def main():
    # reset pause key listener
    win32api.GetAsyncKeyState(pause_key)
    iterations = 0
    state_dset = []
    action_dset = []
    frametimes = deque(maxlen=fps_limit * fps_check_interval)
    frametimes.append(time.time())

    with h5py.File(filepath) as f:
        assert state_name in f and action_name in f
        state_dset = f[state_name][:]
        action_dset = f[action_name][:]

    for state, action in zip(state_dset, action_dset):
        if win32api.GetAsyncKeyState(pause_key):
            print('\nstopped')
            cv2.destroyAllWindows()
            break

        state = cv2.resize(state, preview_size)
        state = draw_angle(state, action[0])
        cv2.imshow('preview', state)
        print(action, end='\r')

        if (cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()

        iterations += 1
        if ((1 / fps_limit) >
                (time.time() - frametimes[-1]) + fps_limit_treshold):
            time.sleep((1 / fps_limit) - (time.time() - frametimes[-1]))
        frametimes.append(time.time())


main()
