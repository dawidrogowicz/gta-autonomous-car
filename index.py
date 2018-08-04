import numpy as np
import time
from PIL import ImageGrab
import cv2
from directkeys import PressKey, ReleaseKey, Keys

is_running = True
window_size = (0, 30, 800, 600)


def roi(img, verts):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, verts, 255)
    out = cv2.bitwise_and(img, mask)
    return out


def draw_lines(img, lines):
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]),
                 (255, 255, 255), 3)


def process_img(img):
    verticies = np.array([[10, 500], [10, 300], [300, 160],
                         [500, 160], [800, 300], [800, 500]])
    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = cv2.Canny(out, threshold1=200, threshold2=300)
    out = roi(out, [verticies])

    lines = cv2.HoughLinesP(cv2.GaussianBlur(out, (1, 3), 0), 1, np.pi / 180,
                            160, np.array([]), 46, 3)
    if lines is not None:
        draw_lines(out, lines)

    return out


def main():
    start = time.time()
    iterations = 0

    while(is_running):
        prtsc = np.array(ImageGrab.grab(bbox=window_size))
        screen = process_img(prtsc)
        iterations += 1

        cv2.imshow('preview', screen)

        if (cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            total = time.time() - start

            print('time: {}\niterations: {}\nfps: {}'
                  .format(total, iterations, iterations / total))
            break


main()
