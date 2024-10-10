from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSlider, QHBoxLayout, QLabel, QWidget


class SliderWithValueLabel(QWidget):
    def __init__(self, minimum=0, maximum=100, value=0,  orientation=Qt.Orientation.Horizontal, label_position='right', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.slider = QSlider(orientation)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)

        self.value_label = QLabel()
        self.label_position = label_position
        self.layout = QHBoxLayout()

        self.valueChanged = self.slider.valueChanged  # connect the value changed signal of the slider to the valueChanged signal of the widget
        self.valueChanged.connect(self._setLabelText)
        self.slider.setValue(value)

        if self.label_position == 'left':
            self.layout.addWidget(self.value_label)

        self.layout.addWidget(self.slider)

        if self.label_position == 'right':
            self.layout.addWidget(self.value_label)

        self.setLayout(self.layout)

    def setRange(self, minValue, maxValue):
        self.slider.setRange(minValue, maxValue)

    def setTickInterval(self, ti):
        self.slider.setTickInterval(ti)

    def setValue(self, value):
        self.slider.setValue(value)
        self.value_label.setText(str(value))

    def _setLabelText(self, value):
        self.value_label.setText(str(value))

    def value(self):
        return self.slider.value()

    def setLabelPosition(self, position):
        if self.label_position == position:
            return

        self.label_position = position
        self.layout.removeWidget(self.value_label)

        if position == 'left':
            self.layout.insertWidget(0, self.value_label)
        elif position == 'right':
            self.layout.addWidget(self.value_label)
