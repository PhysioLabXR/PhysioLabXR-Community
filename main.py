import sys
from PyQt5 import QtWidgets

# Press the green button in the gutter to run the script.
from MainWindow import MainWindow
from interfaces.OpenBCIInterface import OpenBCIInterface

if __name__ == '__main__':
    # Define the sensor interfaces
    eeg_interface = OpenBCIInterface()

    # load the qt application
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(eeg_interface=eeg_interface)
    window.show()
    app.exec_()
    print('Resuming Console Interaction.')

