import time


class Timer:
    def __init__(self):
        self.running_timer = None
        self.root_timers = []

    def tic(self, message):
        if self.running_timer is not None:
            subs_to_check = self.running_timer.subtimers
        else:
            subs_to_check = self.root_timers

        timer_to_start = None
        for sub in subs_to_check:
            if sub.name == message:
                timer_to_start = sub

        if timer_to_start is None:
            timer_to_start = SubTimer(message)
            if self.running_timer is not None:
                self.running_timer.sub(timer_to_start)
            else:
                self.root_timers.append(timer_to_start)

        timer_to_start.start()
        self.running_timer = timer_to_start

    def toc(self):
        if self.running_timer is not None:
            self.running_timer.stop()
            self.running_timer = self.running_timer.parent

    def print(self):
        sorted_subs = sorted(self.root_timers, key=lambda s: s.total, reverse=True)
        for sub in sorted_subs:
            sub.print(0)


class SubTimer:
    def __init__(self, name):
        self.running = False
        self.start_time = 0.0
        self.name = name
        self.min = None
        self.max = 0.0
        self.total = 0.0
        self.n = 0
        self.subtimers = []
        self.parent = None

    def start(self):
        self.start_time = time.clock()
        self.running = True

    def stop(self):
        dt = time.clock() - self.start_time
        self.running = False
        self.total += dt
        self.n += 1
        if self.n == 1:
            self.min = dt
        else:
            self.min = min(self.min, dt)
        self.max = max(self.max, dt)

    def sub(self, sub):
        sub.parent = self
        self.subtimers.append(sub)

    def print(self, indentation):
        print(f"{'': <{indentation*4}}{self.name: <{70-indentation*4}} | sum {self.total*1000: 8.2f} | avg {self.total/self.n*1000: 5.2f} | min {self.min*1000: 5.2f} | max {self.max*1000: 5.2f}")

        sorted_subs = sorted(self.subtimers, key=lambda s: s.total, reverse=True)
        for sub in sorted_subs:
            sub.print(indentation+1)