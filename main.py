import sys
from PyQt5 import QtWidgets

# Press the green button in the gutter to run the script.
from MainWindow import MainWindow
from interfaces.OpenBCIInterface import OpenBCIInterface
from interfaces.UnityLSLInterface import UnityLSLInterface

if __name__ == '__main__':
    # Define the sensor interfaces
    eeg_interface = OpenBCIInterface()
    unityLSL_inferface = UnityLSLInterface()

    # load the qt application
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(eeg_interface=eeg_interface, unityLSL_inferface=unityLSL_inferface)
    window.show()
    app.exec_()
    print('Resuming Console Interaction.')

