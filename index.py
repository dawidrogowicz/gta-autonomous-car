import numpy as np
import time
from PIL import ImageGrab
import cv2
from directkeys import PressKey, ReleaseKey, Keys
from lines import draw_lines

is_running = True
window_size = (0, 30, 800, 600)


def main():
    start = time.time()
    iterations = 0

    while(is_running):
        screen = np.array(ImageGrab.grab(bbox=window_size))
        draw_lines(screen)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        iterations += 1

        cv2.imshow('preview', screen)

        if (cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            total = time.time() - start

            print('time: {}\niterations: {}\nfps: {}'
                  .format(total, iterations, iterations / total))
            break


main()
