from collections import deque
import threading
import time


class FpsSync:
    """Limits iterations on main thread to a specified fps limit.

    Args:
        limit (int): Target fps value to limit.
        span (int, float, optional): Span of time in seconds
            over which fps are checked. Defaults to 1.
        frametime_treshold (int, float, optional): Fraction of frametime
            treshold, required for frame to be freezed. Defaults to .2.
    """

    def __init__(self, limit, span=1, frametime_treshold=.2):
        self.limit = limit
        self.span = span
        self.treshold = 1 / limit * frametime_treshold
        self.buffer = deque(maxlen=int(limit * span))

    def get_fps(self):
        """Returns current framerate."""

        return len(self.buffer) / (time.time() - self.buffer[0])

    def init(self):
        """Initializes buffer with first framerate."""

        self.buffer.append(time.time())

    def sync(self):
        """Records last frametime and freezes it if necessary."""

        if ((1 / self.limit) > (
                time.time() - self.buffer[-1]) + self.treshold):
            time.sleep((1 / self.limit) - (time.time() - self.buffer[-1]))
        self.buffer.append(time.time())
