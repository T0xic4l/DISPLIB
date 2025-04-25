import time
import logging


class TimeLogger:
    def __init__(self, label):
        self.start = None
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        logging.debug(f"{self.label} - {round(time.perf_counter() - self.start, 2)} seconds")


