import numpy as np
import matplotlib.pyplot as plt
result_buffer = []
length = 10000
value = 0
accumulate_value = 1
decay_factor = 0.99



for i in range(0,length):
    value = (value+accumulate_value)*decay_factor
    result_buffer.append(value)

plt.plot(result_buffer[:100])
plt.show()

print(result_buffer)