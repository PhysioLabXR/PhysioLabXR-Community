import time

from pylsl import StreamInfo, StreamOutlet, cf_float32
import pylsl

while True:
    print(time.time())
    # print the current pylsl time
    # print(pylsl.local_clock())