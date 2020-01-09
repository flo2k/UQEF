import time
import sys

class TimeDuration:
    def __init__(self, text, start):
        self.text = text
        self.start = start
        self.end = None

    def set_end(self, end):
        self.end = end

    def duration(self):
        return self.end - self.start

    def print(self):
        print("{}: takes: {}".format(self.text, self.duration()))


class TimeMeasurement:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        self.time_duration = TimeDuration(self.text, time.time())
        sys.stdout.write(self.text)
        sys.stdout.flush()
        return self.time_duration

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time_duration.set_end(time.time())
        print(" takes: {}".format(self.time_duration.duration()))


#usage
#with TimeMeasurement("Time measurement") as t:
#    time.sleep(1)
#
#t.print()


