from PyQt6.QtCore import QThread, pyqtSignal, QMetaObject, Q_ARG
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QDialog, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

class LongTaskThread(QThread):
    completed = pyqtSignal()

    def __init__(self, parent, func_name: str):
        super().__init__(parent)
        self.parent = parent
        self.func_name =func_name

    def run(self):
        QMetaObject.invokeMethod(self.parent, self.func_name, Qt.AutoConnection, Q_ARG(QWidget, self.parent))
        self.completed.emit()

class LoadingDialog(QDialog):
    def __init__(self, parent=None, message="Loading"):
        super().__init__(parent)
        self.message= message

        self.setWindowTitle(message)
        layout = QVBoxLayout()
        self.label = QLabel(message)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_counter = 0

        self.animation_timer.start(500)  # Update animation every 500 milliseconds

    def update_animation(self):
        self.animation_counter = (self.animation_counter + 1) % 4
        dots = "." * self.animation_counter
        self.label.setText(f"{self.message} {dots}")