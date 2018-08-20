import pygame
from pygame import joystick, event


class JoyOutput:
    """Captures output of Joystick controllers.

    Args:
        device_num (int, optional): Number of joystick device. Defaults to 0.
        calibrate_axes ((<int>, ...), optional): Tuple of axes to calibrate.
    """

    def __init__(self, device_num=None, calibrate_axes=()):
        pygame.init()
        self.joy = (joystick.Joystick(device_num)
                    if device_num is not None else self.detect_joy())
        self.joy.init()
        self.offsets = {}
        for i in calibrate_axes:
            self.calibrate(i)

    def calibrate(self, axis=0):
        """Captures real range of inclinations of controller for axis.
        Args:
            axis (int, optional): Index of axis to calibrate. Defaults to 0.
        """

        print('Move your axis all the way to the left and press any button...')
        left = 0
        right = 0
        while True:
            if pygame.JOYBUTTONDOWN in [e.type for e in event.get()]:
                left = self.joy.get_axis(axis)
                break
            print(self.joy.get_axis(axis), end='\r')
        print('Now to the right...')
        while True:
            if pygame.JOYBUTTONDOWN in [e.type for e in event.get()]:
                right = self.joy.get_axis(axis)
                break
            print(self.joy.get_axis(axis), end='\r')

        print('left: {}\nright: {}\nIs that correct?'.format(left, right))
        choice = input('Type `y` for yes, anything else to try again')
        if choice == 'y':
            self.offsets[axis] = {
                'left': left,
                'right': right,
            }
        else:
            self.calibrate(axis)

    def mapOffset(self, axis, val):
        """Maps value of axis from controllers range to max range.

        Args:
            axis (int): Index of axis to map.
            val (int, float): Inclination value of axis.
        """

        assert axis in self.offsets
        left = self.offsets[axis]['left']
        right = self.offsets[axis]['right']
        if val == left:
            return -1
        return (val - left) / (right - left) * 2 - 1

    def detect_joy(self):
        """Asks user to select joystick device"""

        joysticks = [joystick.Joystick(i) for i in range(joystick.get_count())]

        for i, joy in enumerate(joysticks):
            print('{} - {}'.format(i, joy.get_name()))

        choice = input('\nPlease select your controller...')

        if int(choice) not in range(len(joysticks)):
            print('Index out of range, try again')
            return self.detect_joy()

        return joysticks[int(choice)]

    def get_axis(self, axis=0):
        """Returns inclination value of x axis.

        Args:
            axis (int, optional): Axis number. Defaults to 1.
        """

        event.get()
        value = self.joy.get_axis(axis)
        if axis in self.offsets:
            value = self.mapOffset(axis, value)
            if value > 1:
                value = 1.0
        return value
