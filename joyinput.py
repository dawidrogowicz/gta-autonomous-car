import pyvjoy
import time

MAX_JOY_VAL = 32767
j = pyvjoy.VJoyDevice(1)


def centerXAxis():
    j.data.wAxisX = int(MAX_JOY_VAL / 2)
    j.update()


def setXAxis(val, hold_time=False):
    # val ranges from -1 to 1
    j.data.wAxisX = int((val + 1) * MAX_JOY_VAL / 2)
    j.update()

    if hold_time:
        time.sleep(hold_time)
        centerXAxis()
