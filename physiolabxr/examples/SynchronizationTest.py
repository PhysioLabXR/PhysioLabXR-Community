import serial
import time

import numpy as np
import matplotlib.pyplot as plt

from physiolabxr.interfaces.OpenBCIInterface import OpenBCIInterface



arduino_port = serial.Serial('COM8', 9600, timeout=1e-2)  # change this to your serial port
time.sleep(2)

# init connection to Unity LSL
unity_synchronization_interface = UnityLSLInterface(lsl_data_type='Unity.SynchronizationTest')
unity_synchronization_interface.connect_sensor()
unity_synchronization_interface.start_sensor()
unity_synchronization_interface.info()

openBCI_interface = OpenBCIInterface()
openBCI_interface.start_sensor()

def read_line_from_serial(port: serial.Serial):
    b = port.readline()
    str_rn = b.decode()
    str_stripped = str_rn.rstrip()
    return str_stripped

def pulse_detect(data_chunk, pulse_channel=1, pulse_threshold = -8000, debounce_steps=50):
    if len(data_chunk) >= 2:
        delta_array = data_chunk[pulse_channel] - np.concatenate([[np.mean(data_chunk[pulse_channel])], data_chunk[pulse_channel][:-1]])

        pulse_indices = []
        debounce_counter = 0
        for i in range(len(delta_array) - 1):
            if delta_array[i + 1] - delta_array[i] < pulse_threshold and debounce_counter == debounce_steps:
                pulse_indices.append(i)
                debounce_counter = 0
            elif debounce_counter < debounce_steps:
                debounce_counter += 1
        return data_chunk[-1][pulse_indices]
        # return data_chunk[-1][np.where(delta_array > eeg_pulse_height)]  # pull the timestamps where the pulse happened
    else:
        return np.empty(shape=(0,))

python_lsl_time_array = []
unity_time_array = []
arduino_time_array = []

eeg_data_array = np.empty(shape=(31, 0))

# clear all buffers
print('Clearing all buffers')
unity_synchronization_interface.inlet.flush()
openBCI_interface.board.get_board_data()
arduino_port.reset_input_buffer()

print('Entering main loop')
while True:
    # read
    try:
        arduino_line = read_line_from_serial(arduino_port)
        unity_data_chunk, python_lsl_timestamps = unity_synchronization_interface.process_frames()
        eeg_data_chunk = openBCI_interface.process_frames()
        eeg_data_array = np.concatenate((eeg_data_array, eeg_data_chunk), axis=-1)
        if len(unity_data_chunk) > 0:  # Unity issues photon pulse in HMD and let python know through LSL
            print('Python LSL reveived pulse at ' + str(python_lsl_timestamps[0]))
            print('Unity sent pulse at ' + str(unity_data_chunk[0][0]))
            python_lsl_time_array.append(python_lsl_timestamps[0])
            unity_time_array.append(unity_data_chunk[0][0])

        if arduino_line == 'D':  # from Arduino photocell detecting photon pulse from the HMD
            arduino_timestamp = time.time()
            print('Arduino detected pulse at ' + str(arduino_timestamp))
            arduino_time_array.append(arduino_timestamp)
    except KeyboardInterrupt:
        break

eeg_time_array = pulse_detect(eeg_data_array)

print('number of eeg array detected is ' + str(len(eeg_time_array)))

array_p = np.array(python_lsl_time_array) - python_lsl_time_array[0]
array_u = np.array(unity_time_array) - unity_time_array[0]
array_a = np.array(arduino_time_array) - arduino_time_array[0]
array_e = np.array(eeg_time_array) - eeg_time_array[0]

print('Interfaces discrepency    Mean    STD')
p_u_discrepency_mean = np.mean(array_p - array_u)
p_u_discrepency_std = np.std(array_p - array_u)
print('Unity - Python LSL: ' + str(round(p_u_discrepency_mean, 8)) + ' ' + str(round(p_u_discrepency_std, 8)))
u_a_discrepency_mean = np.mean(array_u - array_a)
u_a_discrepency_std = np.std(array_u - array_a)
print('Unity - Arduino Photocell: ' + str(round(u_a_discrepency_mean, 8)) + ' ' + str(round(u_a_discrepency_std, 8)))
a_e_discrepency_mean = np.mean(array_a - array_e)
a_e_discrepency_std = np.std(array_a - array_e)
print('Arduino Photocell - EEG pulse: ' + str(round(a_e_discrepency_mean, 8)) + ' ' + str(round(a_e_discrepency_std, 8)))

plt.plot(eeg_time_array)
plt.show()