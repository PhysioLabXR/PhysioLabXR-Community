# This Python file uses the following encoding: utf-8

# This Python file uses the following encoding: utf-8
from PyQt5 import QtWidgets, uic

from rena.shared import FilterType
from rena.ui.FilterComponentButterworthBandPass import FilterComponentButterworthBandPass


class FilterWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("ui/FilterWidget.ui", self)
        self.filter_widgets = []
        self.addFilterBtn.clicked.connect(self.add_filter_btn_clicked)

    def add_filter_btn_clicked(self):
        """
        add inactive filer widget and RenaFilter
        """
        filter_type = self.filterSelectionCombobox.currentText()

        # create filter


        if filter_type == FilterType.ButterworthBandPass:
            filter_widget = FilterComponentButterworthBandPass()
            self.FilterScrollAreaWidgetContents.addWidget(filter_widget)



