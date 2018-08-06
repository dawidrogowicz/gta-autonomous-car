import numpy as np
from math import sqrt, pow, atan, fabs
import cv2


def get_len(x):
    return sqrt(pow(x[2] - x[0], 2) + pow(x[3] - x[1], 2))


def get_tan(x):
    return (x[3] - x[1]) / (x[2] - x[0]) if (x[2] - x[0]) != 0 else 1


def is_similar_angle(x, y, treshold):
    return fabs(atan(get_tan(x)) - atan(get_tan(y))) * 2 / np.pi \
            < treshold


def distance(line, point):
    X = line[2] - line[0]
    Y = line[3] - line[1]
    u = ((X * (point[0] - line[0]) + Y * (point[1] - line[1])) /
         (pow(X, 2) + pow(Y, 2)))

    if u <= 0:
        closest_point = (line[0], line[1])
    elif u >= 1:
        closest_point = (line[2], line[3])
    else:
        closest_point = [line[0] + u * X, line[1] + u * Y]

    return sqrt(pow(point[0] - closest_point[0], 2) +
                pow(point[1] - closest_point[1], 2))


def is_close(x, y, treshold):
    if get_len(x) > get_len(y):
        return distance(x, (y[0], y[1])) < treshold or \
                distance(x, (y[2], y[3])) < treshold
    else:
        return distance(y, (x[0], x[1])) < treshold or \
                distance(y, (x[2], x[3])) < treshold


def merge_lines(lines):
    if len(lines) < 2:
        return lines

    out = [lines[0]]
    ang_treshold = 0.1
    dist_treshold = 20

    for x in lines:
        if tuple(x) in set([tuple(x) for x in out]):
            continue

        is_unique = True
        biggest = x
        for y in range(len(out)):
            if (is_similar_angle(x, out[y], ang_treshold)
                    and is_close(x, out[y], dist_treshold)
                    and not np.array_equal(x, out[y])):
                    is_unique = False
                    out[y] = biggest if get_len(biggest) > \
                        get_len(out[y]) else out[y]

        if is_unique:
            out.append(x)
    out = set([tuple(x) for x in out])
    return out


def roi(img, verts):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, verts, 255)
    out = cv2.bitwise_and(img, mask)
    return out


def get_lines(img):
    verticies = np.array([[10, 500], [10, 300], [300, 160],
                         [500, 160], [800, 300], [800, 500]])
    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = cv2.Canny(img, 200, 300)
    out = roi(out, [verticies])
    lines = cv2.HoughLinesP(cv2.GaussianBlur(out, (1, 3), 0), 1, np.pi / 180,
                            160, np.array([]), 30, 8)
    if lines is not None:
        lines = np.reshape(lines, (len(lines), -1))
        lines = merge_lines(lines)
        lines = list(sorted(lines, key=lambda x: -get_len(x)))[:2]

    return lines


def draw_lines(img):
    lines = get_lines(img)

    if lines is not None:
        for line in lines:
            cv2.line(img, (line[0], line[1]), (line[2], line[3]),
                     (0, 255, 0), 6)
