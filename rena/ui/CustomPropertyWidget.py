from PyQt5 import QtWidgets, uic


class CustomPropertyWidget(QtWidgets.QWidget):
    def __init__(self, parent, property_name, property_value):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.parent = parent
        self.ui = uic.loadUi("ui/CustomPropertyWidget.ui", self)

        self.set_property_label(property_name)
        self.set_property_label(property_value)

    def set_property_label(self, label_name):
        self.PropertyLabel.text = label_name

    def set_property_value(self, value):
        self.PropertyLineEdit.text = value