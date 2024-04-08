import numpy as np
import os
import PySpin
import matplotlib.pyplot as plt
import sys
import time

from physiolabxr.scripting.RenaScript import RenaScript


class PointGreySpinnaker(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

        self.node_bufferhandling_mode = None
        self.sNodemap = None
        self.cam = None

    # Start will be called once when the run button is hit.
    def init(self):
        # Retrieve singleton reference to system object
        system = PySpin.System.GetInstance()

        # Get current library version
        version = system.GetLibraryVersion()
        print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

        # Retrieve list of cameras from the system
        cam_list = system.GetCameras()

        num_cameras = cam_list.GetSize()

        print('Number of cameras detected: %d' % num_cameras)

        # Finish if there are no cameras
        if num_cameras == 0:
            # Clear camera list before releasing system
            cam_list.Clear()
            # Release system instance
            system.ReleaseInstance()
            print('Not enough cameras!')
            self.cam = None
            return

        # Run example on each camera
        self.cam = cam_list[0]

        # start of run_single_camera
        self.nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
        # Initialize camera
        self.cam.Init()
        # Retrieve GenICam nodemap
        nodemap = self.cam.GetNodeMap()

        self.sNodemap = self.cam.GetTLStreamNodeMap()

        # Change bufferhandling mode to NewestOnly
        self.node_bufferhandling_mode = PySpin.CEnumerationPtr(self.sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsReadable(self.node_bufferhandling_mode) or not PySpin.IsWritable(self.node_bufferhandling_mode):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve entry node from enumeration node
        self.node_newestonly = self.node_bufferhandling_mode.GetEntryByName('NewestOnly')
        if not PySpin.IsReadable(self.node_newestonly):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve integer value from entry node
        self.node_newestonly_mode = self.node_newestonly.GetValue()

        # Set integer value from entry node as new value of enumeration node
        self.node_bufferhandling_mode.SetIntValue(self.node_newestonly_mode)

        try:
            self.node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsReadable(self.node_acquisition_mode) or not PySpin.IsWritable(self.node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            # Retrieve entry node from enumeration node
            self.node_acquisition_mode_continuous = self.node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsReadable(self.node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            # Retrieve integer value from entry node
            self.acquisition_mode_continuous = self.node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            self.node_acquisition_mode.SetIntValue(self.acquisition_mode_continuous)

            print('Acquisition mode set to continuous...')

            #  Begin acquiring images
            #
            #  *** NOTES ***
            #  What happens when the camera begins acquiring images depends on the
            #  acquisition mode. Single frame captures only a single image, multi
            #  frame catures a set number of images, and continuous captures a
            #  continuous stream of images.
            #
            #  *** LATER ***
            #  Image acquisition must be ended when no more images are needed.
            self.cam.BeginAcquisition()

            print('Acquiring images...')

            #  Retrieve device serial number for filename
            #
            #  *** NOTES ***
            #  The device serial number is retrieved in order to keep cameras from
            #  overwriting one another. Grabbing image IDs could also accomplish
            #  this.
            device_serial_number = ''
            self.node_device_serial_number = PySpin.CStringPtr(self.nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsReadable(self.node_device_serial_number):
                device_serial_number = self.node_device_serial_number.GetValue()
                print('Device serial number retrieved as %s...' % device_serial_number)

            image_result = self.cam.GetNextImage(1000)
            image_data = image_result.GetNDArray()
            image_result.Release()
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

    # loop is called <Run Frequency> times per second
    def loop(self):
        print('Loop function is called')
        if self.cam is None:
            return False
        try:
            #  Retrieve next received image
            #
            #  *** NOTES ***
            #  Capturing an image houses images on the camera buffer. Trying
            #  to capture an image that does not exist will hang the camera.
            #
            #  *** LATER ***
            #  Once an image from the buffer is saved and/or no longer
            #  needed, the image must be released in order to keep the
            #  buffer from filling up.

            image_result = self.cam.GetNextImage(1000)

            #  Ensure image completion
            if image_result.IsIncomplete():
                print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
            else:
                # Getting the image data as a numpy array
                image_data = image_result.GetNDArray()

            # send the image datat to the output
            self.outputs["PointGreyEyeCamera"] = np.ravel(image_data)
            #  Release image
            #  *** NOTES ***
            #  Images retrieved directly from the camera (i.e. non-converted
            #  images) need to be released in order to keep from filling the
            #  buffer.
            image_result.Release()
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
        self.cam.EndAcquisition()
        self.cam.DeInit()
        del self.cam



