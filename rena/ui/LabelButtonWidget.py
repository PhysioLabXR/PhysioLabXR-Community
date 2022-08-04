from PyQt5 import QtWidgets, uic


class LabelButtonWidget(QtWidgets.QWidget):
    def __init__(self, label_text):
        super().__init__()
        self.ui = uic.loadUi("ui/LabelButtonWidget.ui", self)

        self.label.setText(label_text)

    def set_button_callback(self, callback):
        self.button.clicked.connect(callback)

    def set_label_2_text(self, text):
        self.label_2.setText(text)

    def get_label_text(self):
        return self.label.text()