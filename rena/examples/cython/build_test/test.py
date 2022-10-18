

# import time
# import numpy
#
#
# total = 0
# arr = numpy.arange(100000000)
#
# t1 = time.time()
#
# for k in arr:
#     total = total + k
# print("Total = ", total)
#
# t2 = time.time()
# t = t2 - t1
# print("%.20f" % t)
import source

print(source.find_primes_vanilla(40))