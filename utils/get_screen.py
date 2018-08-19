import cv2
import numpy as np
import win32gui
import win32ui
import win32con


def get_screen(win_name=None):
    default_rect = (0, 0, 800, 600)

    if win_name:
        try:
            hwin = win32gui.FindWindow(None, win_name)
            rect = win32gui.GetWindowRect(hwin)
        except Exception:
            hwin = win32gui.GetDesktopWindow()
            rect = default_rect
    else:
        hwin = win32gui.GetDesktopWindow()
        rect = default_rect

    left = 3
    top = 26
    x, y, x2, y2 = rect
    width = x2 - x - 6
    height = y2 - y - top - 3

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
