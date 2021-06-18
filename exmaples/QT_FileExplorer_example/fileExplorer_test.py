import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QFileDialog, QApplication


class filedialogdemo(QWidget):
    def __init__(self, parent=None):
        super(filedialogdemo, self).__init__(parent)

        layout = QVBoxLayout()
        self.btn = QPushButton("QFileDialog static method demo")
        self.btn.clicked.connect(self.getfile)

        layout.addWidget(self.btn)
        self.le = QLabel("Hello")

        layout.addWidget(self.le)
        self.btn1 = QPushButton("QFileDialog object")
        self.btn1.clicked.connect(self.getfiles)
        layout.addWidget(self.btn1)

        self.contents = QTextEdit()
        layout.addWidget(self.contents)
        self.setLayout(layout)
        self.setWindowTitle("File Dialog demo")

    def getfile(self):
        # this is for file selection
        # fname = QFileDialog.getOpenFileName(self, 'John',
        #                                     'c:\\', "Image files (*.jpg *.gif)")

        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        print(file)

    def getfiles(self):
        dlg = QFileDialog()



def main():
    app = QApplication(sys.argv)
    ex = filedialogdemo()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()