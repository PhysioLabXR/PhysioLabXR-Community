import zmq
import msgpack
from pylsl import local_clock
import cv2
import numpy as np

def connect_to_pupil_core():
    # Set up ZeroMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ)  # Request socket

    # Connect to Pupil Core's default localhost address and port
    socket.connect("tcp://127.0.0.1:50020")

    # Subscribe to world camera frames
    socket.send_string("SUB_PORT")
    sub_port = socket.recv_string()

    # Set up subscriber socket for frame data
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(f"tcp://127.0.0.1:{sub_port}")

    # only subscribing to the world frame
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "frame.world")

    print("Connected to Pupil Core on localhost")
    return socket, sub_socket


def get_world_camera_image(sub_socket):
    try:
        while True:
            # Receive multipart message
            message = sub_socket.recv_multipart()

            # Print raw message structure for debugging
            print(f"Received {len(message)} parts:")
            for i, part in enumerate(message):
                print(f"Part {i}: {len(part)} bytes, starts with {part[:10]}")

            # Check we have at least 3 parts
            if len(message) < 3:
                print("Incomplete message, expecting topic, metadata, and image data")
                continue

            # Extract parts
            topic = message[0].decode('utf-8')  # Part 0: Topic
            metadata = msgpack.unpackb(message[1], raw=False)  # Part 1: Metadata
            image_data = message[2]  # Part 2: Image bytes

            # Extract metadata fields
            timestamp = metadata.get('timestamp')
            width = metadata.get('width')
            height = metadata.get('height')
            frame_format = metadata.get('format')

            print(f"New frame received:")
            print(f"Topic: {topic}")
            print(f"Timestamp: {timestamp}")
            print(f"Resolution: {width}x{height}")
            print(f"Format: {frame_format}")

            # Visualize the image
            if image_data and frame_format == 'jpeg':
                print("Attempting to decode JPEG image...")
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imshow('World Camera', img)
                    cv2.waitKey(1)
                else:
                    print("Failed to decode image data as JPEG")
            else:
                print("No valid image data or unsupported format")
                print(f"image_data exists: {bool(image_data)}, format match: {frame_format == 'jpeg'}")

            # time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    '''
    Todo:   
    '''
    # start pupil lab capture executable

    # Connect to Pupil Core
    socket, sub_socket = connect_to_pupil_core()

    # Start Pupil Core time synchronization (optional)
    system_time =local_clock()
    socket.send_string(f"T {system_time}")
    response = socket.recv_string()
    print("Timesync successful." if response == "Timesync successful." else response)

    # Get world camera images
    print("Starting to receive world camera frames...")
    get_world_camera_image(sub_socket)