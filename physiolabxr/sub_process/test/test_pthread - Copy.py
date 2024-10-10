import time

from PyQt6.QtCore import pyqtSlot, QTimer
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QApplication,
)


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setFixedSize(250, 100)
        self.setWindowTitle("Sheep Picker")

        self.sheep_number = 1
        self.timer = QTimer()
        self.picked_sheep_label = QLabel()
        self.counted_sheep_label = QLabel()

        self.layout = QVBoxLayout()
        self.main_widget = QWidget()
        self.pick_sheep_button = QPushButton("Pick a sheep!")

        self.layout.addWidget(self.counted_sheep_label)
        self.layout.addWidget(self.pick_sheep_button)
        self.layout.addWidget(self.picked_sheep_label)

        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)

        self.timer.timeout.connect(self.count_sheep)
        self.pick_sheep_button.pressed.connect(self.pick_sheep)

        self.timer.start()

    @pyqtSlot()
    def count_sheep(self):
        self.sheep_number += 1
        self.counted_sheep_label.setText(f"Counted {self.sheep_number} sheep.")

    @pyqtSlot()
    def pick_sheep(self):
        self.picked_sheep_label.setText(f"Sheep {self.sheep_number} picked!")
        time.sleep(5)  # This function represents a long-running task!


if __name__ == "__main__":
    app = QApplication([])

    main_window = MainWindow()
    main_window.show()

    app.exec()