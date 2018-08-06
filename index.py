import numpy as np
import time
from math import fabs
import cv2
from directkeys import PressKey, ReleaseKey, Keys
from lines import draw_lines, get_tan
from get_screen import get_screen

is_running = True


def turn_left():
    PressKey(Keys['A'])
    ReleaseKey(Keys['W'])
    ReleaseKey(Keys['D'])


def turn_right():
    PressKey(Keys['D'])
    ReleaseKey(Keys['W'])
    ReleaseKey(Keys['A'])
    ReleaseKey(Keys['A'])


def speed_up():
    PressKey(Keys['W'])
    ReleaseKey(Keys['A'])
    ReleaseKey(Keys['D'])


def slow_down():
    ReleaseKey(Keys['W'])


def steer(tan, img):
    if fabs(tan) * 2 > np.pi * .5:
        speed_up()
        cv2.putText(img, '{}: ^'.format(tan), (160, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), cv2.LINE_4)
    else:
        slow_down()
        cv2.putText(img, '{}: -'.format(tan), (160, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), cv2.LINE_4)

    if fabs(tan) * 2 < np.pi * .8:
        if tan < 0:
            turn_right()
            cv2.putText(img, '{}: <'.format(tan), (160, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        cv2.LINE_4)
        else:
            turn_left()
            cv2.putText(img, '{}: >'.format(tan), (160, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        cv2.LINE_4)


def main():
    start = time.time()
    iterations = 0

    while(is_running):
        screen = get_screen('Grand Theft Auto V')
        # screen = get_screen()
        lines, direction = draw_lines(screen)

        if direction is not None:
            steer(get_tan(direction), screen)

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
