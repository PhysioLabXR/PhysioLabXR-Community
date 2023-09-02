import cv2

from physiolabxr.ui.SplashScreen import SplashLoadingTextNotifier


def get_working_camera_id():
    """
    deprecated, not in use. Use the more optimized version as in general.get_working_camera_ports()
    :return:
    """
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr


def get_working_camera_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing.
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            SplashLoadingTextNotifier().set_loading_text("Video port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                SplashLoadingTextNotifier().set_loading_text("Video port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                SplashLoadingTextNotifier().set_loading_text("Video port %s for camera ( %s x %s) is present but does not read." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
        camera.release()
    return available_ports, working_ports, non_working_ports