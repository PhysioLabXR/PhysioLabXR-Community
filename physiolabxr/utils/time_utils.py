import time

try:
    from pylsl import local_clock
    pylsl_imported = True
except:
    pylsl_imported = False

def get_clock_time():
    """Get the current time in seconds since system starts.

    We can get this time either from the system clock or from the LSL clock. The two are similar.

    """
    if pylsl_imported:
        return local_clock()
    else:
        return time.monotonic()