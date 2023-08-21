from PyQt6.QtCore import QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator


class NoCommaIntValidator(QRegularExpressionValidator):
    def __init__(self, minimum=None, maximum=None, parent=None):
        # Create a regular expression that allows only positive or negative integers without commas
        regex = QRegularExpression(r'^-?\d+$')
        super().__init__(regex, parent)
        self.minimum = minimum
        self.maximum = maximum

    def setRange(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum


    def validate(self, input_text, pos):
        result = super().validate(input_text, pos)

        if result[0] == QRegularExpressionValidator.State.Acceptable:
            # Convert the input text to an integer and check against the specified range
            value = int(input_text)
            if self.minimum is not None and value < self.minimum:
                return (QRegularExpressionValidator.State.Invalid, input_text, pos)
            if self.maximum is not None and value > self.maximum:
                return (QRegularExpressionValidator.State.Invalid, input_text, pos)

        return result
