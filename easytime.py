import time


class Timer:
    def __init__(self):
        self.print_enabled = False
        self.tic_time = 0

    def tic(self):
        self.tic_time = time.clock()

    def toc(self):
        dt = time.clock() - self.tic_time
        if self.print_enabled:
            print(dt * 1000)
        return dt

    def enable(self):
        self.print_enabled = True

    def disable(self):
        self.print_enabled = False