import sys
from collections import deque
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class SignalTreeViewWindow(QTreeWidget):
    def __init__(self, parent, preset):
        # super(SignalTreeViewWindow, self).__init__(parent=parent)
        super().__init__()
        self.parent = parent
        self.preset = preset
        # self.model = QStandardItemModel()
        # self.model.setHorizontalHeaderLabels(['Display', 'Name'])

        # self.header().setDefaultSectionSize(180)
        self.setHeaderHidden(True)
        # self.setModel(self.model)
        self.createTreeView()
        self.expandAll()
        # self.show()


    def createTreeView(self):

        for i in range(3):
            parent = QTreeWidgetItem(self)
            parent.setText(0, "Parent {}".format(i))
            parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
            for x in range(5):
                child = QTreeWidgetItem(parent)
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setText(0, "Child {}".format(x))
                child.setCheckState(0, Qt.Unchecked)
