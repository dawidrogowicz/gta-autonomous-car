import pygame
from pygame import joystick, event


class JoyOutput:
    """Captures output of Joystick controllers.

    Args:
        device_num (int, optional): Number of joystick device. Defaults to 0.
    """

    def __init__(self, device_num=None):
        pygame.init()
        self.joy = (joystick.Joystick(device_num)
                    if device_num is not None else self.detect_joy())
        self.joy.init()

    def detect_joy(self):
        print('Press button on your controller to calibrate...')

        joysticks = [joystick.Joystick(i) for i in range(joystick.get_count())]

        for i, joy in enumerate(joysticks):
            print('{} - {}'.format(i, joy.get_name()))

        choice = input('\nPlease select your controller...')

        if int(choice) not in range(len(joysticks)):
            print('Index out of range, try again')
            return self.detect_joy()

        return joysticks[int(choice)]

    def get_axis(self, axis_num=0):
        """Returns inclination value of x axis.

        Args:
            axis_num (int, optional): Axis number. Defaults to 1.
        """

        event.get()
        return self.joy.get_axis(axis_num)
