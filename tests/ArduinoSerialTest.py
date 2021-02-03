import serial
import time

ser = serial.Serial('/dev/tty.usbmodem144101', 9600)
time.sleep(2)

while 1:
    b = ser.readline()
    str_rn = b.decode()
    str_stripped = str_rn.rstrip()
    if str_stripped:
        print('read:' + str_stripped)