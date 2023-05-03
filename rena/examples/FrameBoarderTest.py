import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QFrame


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        frame = QFrame(self)
        frame.setGeometry(50, 50, 200, 200)
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)

        vbox = QVBoxLayout()
        vbox.addWidget(frame)

        self.setLayout(vbox)
        self.setGeometry(300, 300, 300, 300)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())