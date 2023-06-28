from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QSplitter, QSlider
from pyqtgraph import PlotWidget, mkPen
import numpy as np

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Create the two plot widgets
        self.plot1 = PlotWidget()
        self.plot2 = PlotWidget()
        self.plot3 = PlotWidget()

        # Set the pen colors for the plots
        self.plot1.plot(np.sin(np.linspace(0, 2*np.pi, 100)), pen=mkPen('r', width=2))
        self.plot2.plot(np.cos(np.linspace(0, 2*np.pi, 100)), pen=mkPen('b', width=2))
        self.plot3.plot(np.cos(np.linspace(0, 2*np.pi, 100)), pen=mkPen('b', width=2))

        # Create the splitter
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.plot1)
        self.splitter.addWidget(self.plot2)
        self.splitter.addWidget(self.plot3)


        # # Create the slider
        # self.slider = QSlider(Qt.Vertical)
        # self.slider.setMinimum(1)
        # self.slider.setMaximum(99)
        # self.slider.setValue(50)
        # self.slider.setTickInterval(10)
        # self.slider.setTickPosition(QSlider.TicksBothSides)
        # self.slider.valueChanged.connect(self.resize_plots)

        # Create the main layout and add the splitter and slider
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.splitter)
        # self.layout.addWidget(self.slider)

        # Set the main layout for the window
        self.setLayout(self.layout)

    # def resize_plots(self, value):
    #     # Calculate the height of each plot widget based on the slider value
    #     plot1_height = (value / 100) * self.splitter.height()
    #     plot2_height = self.splitter.height() - plot1_height
    #
    #     # Set the sizes of the plot widgets
    #     self.splitter.widget(0).setFixedSize(self.splitter.width(), plot1_height)
    #     self.splitter.widget(1).setFixedSize(self.splitter.width(), plot2_height)


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()