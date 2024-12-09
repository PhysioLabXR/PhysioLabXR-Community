import os
import ctypes
import numpy as np
import time

dll_path = os.path.join(os.path.dirname(__file__), 'DWT.dll')
dwt_dll = ctypes.CDLL(dll_path)
dwt_dll.find_cost.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t
]
dwt_dll.find_cost.restype = ctypes.c_double

s1 = np.array([[1.0, 2.0], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1], [2.3, 1.1]], dtype=np.float64)
s2 = np.array([[1.0, 2.0], [1.1, 1.5], [2.6, 1.12], [2.1, 1.12]], dtype=np.float64)
print(s1.shape)
start = time.perf_counter()
for i in range(15812):
    s1[0][0] += 5.0
    s2[0][0] += 1.1
    cost = dwt_dll.find_cost(
        s1.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        s1.shape[0],
        s2.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        s2.shape[0],
    )
print(time.perf_counter() - start)
print(cost)