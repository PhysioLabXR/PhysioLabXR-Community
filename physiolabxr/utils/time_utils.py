import time
from datetime import datetime

try:
    from pylsl import local_clock
    pylsl_imported = True
except:
    pylsl_imported = False

def get_clock_time():
    if pylsl_imported:
        return local_clock()
    else:
        return time.monotonic()


def get_datetime_str():
    return datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
