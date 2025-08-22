import zmq
import msgpack
from pylsl import local_clock
import cv2
import numpy as np

def connect_to_pupil_core(context):
    # Set up ZeroMQ context and socket
    socket = context.socket(zmq.REQ)  # Request socket

    # Connect to Pupil Core's default localhost address and port
    socket.connect("tcp://127.0.0.1:50020")
    # Subscribe to world camera frames
    socket.send_string("SUB_PORT")
    sub_port = socket.recv_string()

    # Set up subscriber socket for frame data
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(f"tcp://127.0.0.1:{sub_port}")
    sub_socket.setsockopt(zmq.CONFLATE, 1)

    # only subscribing to the world frame
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "frame.world")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "pupil.")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "gaze.")

    print("Connected to Pupil Core on localhost")
    return socket, sub_socket


def get_world_camera_image(socket_sub, socket_world_pub, socket_eye_pub, socket_pupil_l, socket_pupil_r,
                           topic_world_camera_pub="PupilLabsWorldCamera",
                           topic_eye_pub="PupilLabsEye",
                           topic_epupil_l_pub="PupilLabsPupilL",
                           topic_epupil_r_pub="PupilLabsPupilR"):
    try:
        while True:
            # Receive multipart message
            message = socket_sub.recv_multipart()

            # Print raw message structure for debugging
            # print(f"Received {len(message)} parts:")
            # for i, part in enumerate(message):
            #     print(f"Part {i}: {len(part)} bytes, starts with {part[:10]}")

            # Check we have at least 3 parts


            # Extract parts
            topic = message[0].decode('utf-8')  # Part 0: Topic
            metadata = msgpack.unpackb(message[1], raw=False)  # Part 1: Metadata
            timestamp = metadata.get("timestamp")

            if topic.startswith("frame.world"):
                if len(message) < 3:
                    print("Incomplete message, expecting topic, metadata, and image data")
                    continue
                image_data = message[2]
                frame_format = metadata.get("format")

                if image_data and frame_format == "jpeg":
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    print("No valid world camera image data.")
                    img = None

                socket_world_pub.send_multipart([bytes(topic_world_camera_pub, "utf-8"),
                                           np.array(timestamp),
                                           np.reshape(img, -1)])
            else:
                # channels are [binocular x, binocular y, left pupil, right pupil]
                if topic.startswith("pupil."):
                    frame = np.array([metadata.get("diameter")])
                    if metadata.get("id") == 0:
                        socket_pupil_l.send_multipart([bytes(topic_epupil_l_pub, "utf-8"), np.array(timestamp), frame])
                    else:
                        socket_pupil_r.send_multipart([bytes(topic_epupil_r_pub, "utf-8"), np.array(timestamp), frame])
                elif topic.startswith("gaze.3d.01."):
                    gaze_x, gaze_y = metadata.get("norm_pos", (None, None))
                    frame = np.array([gaze_x, gaze_y])
                    socket_eye_pub.send_multipart([bytes(topic_eye_pub, "utf-8"), np.array(timestamp), frame])

    except KeyboardInterrupt:
        print("\nStopped by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # TODO start pupil lab capture executable
    context = zmq.Context()
    port_world_camera = 5557
    port_eye = 5558
    port_pupil_l = 5559
    port_pupil_r = 5560

    socket_world_camera_pub = context.socket(zmq.PUB)
    socket_world_camera_pub.bind("tcp://*:%s" % port_world_camera)
    # socket_world_camera_pub.setsockopt(zmq.CONFLATE, 1)

    socket_eye_pub = context.socket(zmq.PUB)
    socket_eye_pub.bind("tcp://*:%s" % port_eye)
    # socket_eye_pub.setsockopt(zmq.CONFLATE, 1)

    socket_pupil_l = context.socket(zmq.PUB)
    socket_pupil_l.bind("tcp://*:%s" % port_pupil_l)
    # socket_pupil_l.setsockopt(zmq.CONFLATE, 1)

    socket_pupil_r = context.socket(zmq.PUB)
    socket_pupil_r.bind("tcp://*:%s" % port_pupil_r)
    # socket_pupil_r.setsockopt(zmq.CONFLATE, 1)

    # Connect to Pupil Core
    socket_req, socket_sub = connect_to_pupil_core(context)

    # Start Pupil Core time synchronization (optional)
    system_time =local_clock()
    socket_req.send_string(f"T {system_time}")
    response = socket_req.recv_string()
    print("Timesync successful." if response == "Timesync successful." else response)

    # Get world camera images
    print("Starting to receive world camera frames...")
    get_world_camera_image(socket_sub, socket_world_camera_pub, socket_eye_pub, socket_pupil_l, socket_pupil_r)