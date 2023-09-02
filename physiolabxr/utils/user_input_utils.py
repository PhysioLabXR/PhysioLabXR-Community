from PyQt6.QtCore import QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator

float_validator = QRegularExpressionValidator(QRegularExpression(r"[-+]?[0-9]*\.?[0-9]+"))