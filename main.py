import sys

from PyQt5 import QtWidgets

# Press the green button in the gutter to run the script.
from MainWindow import MainWindow
from interfaces.InferenceInterface import InferenceInterface

if __name__ == '__main__':
    # Define the sensor interfaces
    inference_interface = InferenceInterface()

    # load the qt application
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(inference_interface=inference_interface)
    window.show()
    app.exec_()
    print('Resuming Console Interaction.')

