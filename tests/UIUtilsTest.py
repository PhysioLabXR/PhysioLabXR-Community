import sys
from enum import Enum

from PyQt6 import QtWidgets

from physiolabxr.utils.ui_utils import add_enum_values_to_combobox


def test_add_enum_values_to_combobox():
    """Test case for the function utils.ui_utils.add_enum_values_to_combobox

    The test does the following
    1. create a dummy pyqt app with a combobox
    2. call add_enum_values_to_combobox with a dummy enum class
    3. check if the combobox has the correct number of items
    4. check if the combobox has the correct items (i.e., name of the enums)
    5. check if we can set the CurrentIndex of the combobox by the name of the enum through findtext
    """

    class DummyEnum(Enum):
        A = 1
        B = 2
        C = 3

    app = QtWidgets.QApplication(sys.argv)
    combobox = QtWidgets.QComboBox()
    add_enum_values_to_combobox(combobox, DummyEnum)
    assert combobox.count() == 3
    assert combobox.itemText(0) == "A"
    assert combobox.itemText(1) == "B"
    assert combobox.itemText(2) == "C"
    assert combobox.findText("A") == 0
    assert combobox.findText("B") == 1
    assert combobox.findText("C") == 2

    item_index = combobox.findText("A")
    combobox.setCurrentIndex(item_index)

    assert combobox.currentIndex() == 0

    app.quit()