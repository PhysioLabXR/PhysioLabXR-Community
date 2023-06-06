import sys

from rena.interfaces.DeviceInterface import DeviceInterface
from rena.utils.ConfigPresetUtils import DeviceType
import tobii_research as tr


class RenaAudioInputInterface(DeviceInterface):

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
            print("Address: " + eyetracker.address)
            print("Model: " + eyetracker.model)
            print("Name (It's OK if this is empty): " + eyetracker.device_name)
            print("Serial number: " + eyetracker.serial_number)

    # setup the global variables
        last_report = 0
        N = 0
        halted = False

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

    def gase_data_callback(self, gaze_data):
        try:
            global last_report
            global outlet
            global N
            global halted

            sts = gaze_data['system_time_stamp'] / 1000000.

            outlet.push_sample(self.unpack_gaze_data(gaze_data), sts)

            if sts > last_report + 5:
                sys.stdout.write("%14.3f: %10d packets\r" % (sts, N))
                last_report = sts
            N += 1

            # print(unpack_gaze_data(gaze_data))
        except:
            print("Error in callback: ")
            print(sys.exc_info())

            halted = True

    def start_gaze_tracking(self):
        self.eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
        return True


if __name__ == '__main__':
    print()
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
