import numpy as np
import time
import cv2
import os
import sys
import win32api
from utils.get_screen import get_screen
from utils.joyinput import JoyInput
from utils.fps_sync import FpsSync
from keras.models import Sequential, load_model

is_running = True
fps_limit = 16
model_path_template = 'models/prototype.{id}.hdf5'
# 27 refers to ESC key
pause_key = 27
# openCV uses shape: (height, width)
state_dim = (60, 80)


def play(model_id):
    model_path = model_path_template.format(id=model_id)
    assert model_id and os.path.isfile(model_path)
    # reset pause key listener
    win32api.GetAsyncKeyState(pause_key)
    iterations = 0
    fps_sync = FpsSync(fps_limit)
    model = load_model(model_path)
    joy = JoyInput()

    for i in range(3)[::-1]:
        print('starting in {} seconds'.format(i), end='\r')
        time.sleep(1)
    print('\nPlaying!')

    fps_sync.init()

    while(is_running):
        if win32api.GetAsyncKeyState(pause_key):
            print('\nstopped')
            break

        state = get_screen('Grand Theft Auto V')
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state, state_dim[::-1])
        state = np.expand_dims(state, -1)
        state = np.expand_dims(state, 0)
        action = model.predict(state, batch_size=1)

        joy.set_x_axis(action[0])

        iterations += 1
        if iterations % 10 == 0:
            print('fps: {}'.format(fps_sync.get_fps()), end='\r')

        fps_sync.sync()


if __name__ == '__main__':
    assert len(sys.argv) > 1
    model_id = sys.argv[1]
    play(model_id)
