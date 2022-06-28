import sys
from collections import deque
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class SignalTreeViewWindow(QTreeView):
    def __init__(self, parent):
        super(SignalTreeViewWindow, self).__init__(parent=parent)
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['Display', 'Name'])
        self.header().setDefaultSectionSize(180)
        self.setModel(self.model)
        self.importData(data=None)
        self.expandAll()


    def importData(self, data, root=None):
        if root is None:
            root = self.model.invisibleRootItem()

        seen = {}   # List of  QStandardItem
        
