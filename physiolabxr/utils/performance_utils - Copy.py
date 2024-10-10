import time

def timeit(function: callable, args):
    """
    time a function with performance counter.
    @param function:
    @param args:
    @return: what the given function originally returns plus the run time
    """
    start_time = time.perf_counter()
    returns = function(*args)
    return returns, time.perf_counter() - start_time