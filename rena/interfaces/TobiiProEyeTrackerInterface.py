import subprocess
import sys

from rena.interfaces.DeviceInterface import DeviceInterface
from rena.utils.ConfigPresetUtils import DeviceType
import tobii_research as tr
import time
import random
import os
import pylsl as lsl
import sys

class TobiiProEyeTrackerInterface(DeviceInterface):

    def __init__(self, _device_name, _device_type: DeviceType):
        super().__init__(_device_name, _device_type)
        self.channel_num = 31

        self.gaze_stuff = [
            ('device_time_stamp', 1),

            ('left_gaze_origin_validity', 1),
            ('right_gaze_origin_validity', 1),

            ('left_gaze_origin_in_user_coordinate_system', 3),
            ('right_gaze_origin_in_user_coordinate_system', 3),

            ('left_gaze_origin_in_trackbox_coordinate_system', 3),
            ('right_gaze_origin_in_trackbox_coordinate_system', 3),

            ('left_gaze_point_validity', 1),
            ('right_gaze_point_validity', 1),

            ('left_gaze_point_in_user_coordinate_system', 3),
            ('right_gaze_point_in_user_coordinate_system', 3),

            ('left_gaze_point_on_display_area', 2),
            ('right_gaze_point_on_display_area', 2),

            ('left_pupil_validity', 1),
            ('right_pupil_validity', 1),

            ('left_pupil_diameter', 1),
            ('right_pupil_diameter', 1)
        ]

        self.eye_tracker = None

    def start_sensor(self):
        eyetracker = tr.find_all_eyetrackers()
        if len(eyetracker) == 0:
            print("No eyetracker found")
            exit(1)
        else:
            print("Found eyetracker with serial number " + eyetracker[0].serial_number)
            self.eye_tracker = eyetracker[0]
            print("Address: " + self.eye_tracker.address)
            print("Model: " + self.eye_tracker.model)
            print("Name (It's OK if this is empty): " + self.eye_tracker.device_name)
            print("Serial number: " + self.eye_tracker.serial_number)
            # set up the global variables
        self.last_report = 0
        self.N = 0
        self.halted = False

        self.outlet = self.setup_lsl()
        print("Starting gaze tracking")
        self.start_gaze_tracking()

    def stop_sensor(self):
        pass

    def process_frames(self):
        pass


    def unpack_gaze_data(self, gaze_data):
        x = []
        for s in self.gaze_stuff:
            d = gaze_data[s[0]]
            if isinstance(d, tuple):
                x = x + list(d)
            else:
                x.append(d)
        x[0] = 0
        # print(f"x: {x}")
        return x

    def gaze_data_callback(self, gaze_data):
        try:
            # global last_report
            # global N
            # global halted

            sts = gaze_data['system_time_stamp'] / 1000000.

            self.outlet.push_sample(self.unpack_gaze_data(gaze_data), sts)

            if sts > self.last_report + 5:
                sys.stdout.write("%14.3f: %10d packets\r" % (sts, self.N))
                self.last_report = sts
            self.N += 1

            # print(unpack_gaze_data(gaze_data))
        except:
            print("Error in callback: ")
            print(sys.exc_info())

            halted = True

    def start_gaze_tracking(self):
        self.eye_tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback, as_dictionary=True)
        return True

    def end_gaze_tracking(self):
        self.eye_tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback)
        return True

    # def calibrate():
    #     print("Starting calibration")
    #     output = subprocess.check_output([tobii_et_manager_path, '\calibrate'])
    #     print(output)
    def setup_lsl(self):
        # global channels
        # global gaze_stuff

        info = lsl.StreamInfo('Tobii', 'ET', self.channel_num, 90, 'float32', self.eye_tracker.address)
        info.desc().append_child_value("manufacturer", "Tobii")
        channels = info.desc().append_child("channels")
        cnt = 0
        for s in self.gaze_stuff:
            if s[1] == 1:
                cnt += 1
                channels.append_child("channel") \
                    .append_child_value("label", s[0]) \
                    .append_child_value("unit", "device") \
                    .append_child_value("type", 'ET')
            else:
                for i in range(s[1]):
                    cnt += 1
                    channels.append_child("channel") \
                        .append_child_value("label", "%s_%d" % (s[0], i)) \
                        .append_child_value("unit", "device") \
                        .append_child_value("type", 'ET')

        outlet = lsl.StreamOutlet(info)
        return outlet


    def halt(self):
        global halted
        halted = True


if __name__ == '__main__':
    tobii_interface = TobiiProEyeTrackerInterface(_device_name="TobiiPro", _device_type=DeviceType.TOBIIPRO)
    tobii_interface.start_sensor()
    while 1:
        tobii_interface.process_frames()

    # LSL example
    # info = pylsl.StreamInfo("AudioStream", "MyData", 1, 44100, pylsl.cf_int16, "myuniqueid")
    # outlet = pylsl.StreamOutlet(info)
    #
    # audio_interface = RenaAudioInputInterface(stream_name="test", audio_device_index=1, channels=1)
    # audio_interface.start_sensor()
    # while 1:
    #     data, timestamps = audio_interface.process_frames()
    #     if len(timestamps)>0:
    #         for index, sample in enumerate(data.T):
    #             outlet.push_sample(sample, timestamp=timestamps[index])
