import serial
import time

ser = serial.Serial('COM8', 9600, timeout=1e-2)  # change this to your serial port
time.sleep(2)

while 1:
    b = ser.readline()
    str_rn = b.decode()
    str_stripped = str_rn.rstrip()
    if str_stripped:
        print('read:' + str_stripped)