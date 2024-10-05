import time
import zmq
import subprocess
import os
from multiprocessing import Process, Event
from physiolabxr.interfaces.DeviceInterface.DeviceInterface import DeviceInterface


def compile_tobii_pro_fusion_process():
    #if user_os == "Windows":
    base_dir = os.path.dirname(__file__)
    c_file = os.path.join(base_dir, "TobiiProFusion_Process.c")
    output_file = os.path.join(base_dir, "tobiiprofusion_process")
    compile_command = ["gcc", c_file, "-o", output_file, "-g", "-Wall"]

    try:
        subprocess.run(compile_command, check=True)
        print("Compilation successful!")
    except subprocess.CalledProcessError:
        print("Compilation failed.")

def run_tobii_pro_fusion_process(port):
    terminate_event = Event()
    eyetracker_process = Process(target=tobiiprofusion_process, args=(terminate_event, port))
    eyetracker_process.start()
    return eyetracker_process, terminate_event

class TobiiProFusionInterface(DeviceInterface):
    def __init__(self,
                 _device_name='TobiiProFusion',
                 _device_type='TOBIIPRO',
                 _device_nominal_sampling_rate=250):
        super(TobiiProFusionInterface, self).__init__(_device_name=_device_name,
                                                      _device_type=_device_type,
                                                      device_nominal_sampling_rate=_device_nominal_sampling_rate,
                                                      is_supports_device_availability=False,
                                                      )

        self.stream_name = _device_name
        self.stream_type = _device_type
        self.eyetracker = None

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)  # Subscriber socket
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe to all topics
        self.socket.connect("tcp://localhost:0")  # Bind to port 0 for an available random port
        self.port = self.socket.getsockopt(zmq.LAST_ENDPOINT).decode("utf-8").split(":")[
            -1]  # Get the randomly binded port number from the socket

        self.terminate_event = Event()
        self.device_process = None
        self.data_process = None
        self.terminate_event = None

    def start_stream(self):
        self.data_process, self.terminate_event = run_tobii_pro_fusion_process(self.port)

    def stop_stream(self):
        self.terminate_event.set()
        self.data_process.join()
        self.data_process = None
        # empty the socket buffer, so that the next time we start the stream, we don't get old data
        while True:  # do this after the process has been terminated
            try:
                self.socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.error.Again:
                break

    def get_sampling_rate(self):
        return self.device_nominal_sampling_rate

    def process_frames(self):
        # Code to receive and process data from the eye tracker
        frames, timestamps, messages = [], [], []
        while True:  # Collect all available data
            try:
                data = self.socket.recv_pyobj(flags=zmq.NOBLOCK)  # Non-blocking receive
                if data['t'] == 'e':
                    messages.append(data['message'])
                    self.stop_stream();
                elif data['t'] == 'd':
                    frames.append(data['frame'])
                    timestamps.append(data['timestamp'])
            except zmq.Again:
                # No more data available, break the loop
                break

        return frames, timestamps, messages

    def is_device_available(self):
        return self.device_available

    def __del__(self):
        """Clean up ZMQ context and sockets.

        Note that you don't need to terminate the device process here, because this is handled in
        the stop_stream method. And stop_stream is called by the DeviceWorker before the interface is destroyed.
        """
        self.socket.close()
        self.context.term()

 # time.perf_counter_ns()

if __name__ == "__main__":
    compile_tobii_pro_fusion_process()
    # Instantiate the device interface
    tobii_pro_fusion_interface = TobiiProFusionInterface()

    # Start the device stream
    tobii_pro_fusion_interface.start_stream()

    try:
        # Continuously process frames from the device in a test loop
        for _ in range(100):  # Run for 100 iterations (or replace with a time-based loop)
            frames, timestamps, messages = tobii_pro_fusion_interface.process_frames()
            if frames:
                print(f"Frames: {frames}")
                print(f"Timestamps: {timestamps}")
            if messages:
                print(f"Messages: {messages}")

            time.sleep(0.1)  # Adjust sleep time to match expected data rate

    except KeyboardInterrupt:
        print("Test interrupted by user.")

    finally:
        # Stop the device stream and clean up resources
        tobii_pro_fusion_interface.stop_stream()
        print("Device stream stopped and resources cleaned up.")