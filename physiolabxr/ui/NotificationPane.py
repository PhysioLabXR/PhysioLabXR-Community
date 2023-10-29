from PyQt6 import QtWidgets, QtCore


class NotificationPane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(NotificationPane, self).__init__(parent)

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._label = QtWidgets.QLabel(self)
        self._label.setWordWrap(True)
        self._label.setAlignment(QtCore.Qt.AlignTop)
        self._label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self._layout.addWidget(self._label)

        self._label.hide()

    def showNotification(self, message):
        self._label.setText(message)
        self._label.show()

    def hideNotification(self):
        self._label.hide()