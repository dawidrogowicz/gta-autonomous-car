import pyvjoy
import time


class JoyInput:
    """Class for manipulating virtual vJoy device.
    Args:
        device_num (int, optional): Number of vJoy device. Defaults to 1.
    """

    MAX_JOY_VAL = 32767
    """Maximum value of inclination each axis can have."""

    def __init__(self, device_num=1):
        self.joy = pyvjoy.VJoyDevice(device_num)

    def centerXAxis(self):
        """Sets x axis to center position."""
        self.joy.data.wAxisX = int(self.MAX_JOY_VAL / 2)
        self.joy.update()

    def setXAxis(self, val, hold_time=False):
        """Sets x axis to a specified value.

        Args:
            val (int, float): Value of inclination ranging from -1 to 1.
            hold_time(int, float, optional): If `hold_time` is set,
                value will be released after specified time in seconds.
        """
        self.joy.data.wAxisX = int((val + 1) * self.MAX_JOY_VAL / 2)
        self.joy.update()

        if hold_time:
            time.sleep(hold_time)
            self.centerXAxis()
