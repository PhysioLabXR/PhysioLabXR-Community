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

        # self.setSelectionMode(self.SingleSelection)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        # self.setDragEnabled(True)
        # self.setAcceptDrops(True)
        # self.setDropIndicatorShown(True)

    def createTreeView(self):
        channel_groups_information = self.preset['GroupChannelsInPlot']
        print(channel_groups_information)
        for group_name in channel_groups_information:
            parent = QTreeWidgetItem(self)
            parent.setText(0, group_name)
            parent.setText(1, group_name)
            parent.setFlags(parent.flags() |
                            Qt.ItemIsTristate | Qt.ItemIsUserCheckable | Qt.ItemIsEditable |
                            Qt.ItemIsDragEnabled)

            for channel_index in channel_groups_information[group_name]['channels']:
                child = QTreeWidgetItem(parent)
                child.setFlags(child.flags()
                # | Qt.ItemIsTristate
                | Qt.ItemIsUserCheckable| Qt.ItemIsEditable
                | Qt.ItemIsDragEnabled
                )
                child.setText(0, self.preset['ChannelNames'][channel_index])
                child.setCheckState(0, Qt.Unchecked)

    def startDrag(self, actions):
        # row = self.selectedItems()[0]
        # prev_parent = row.parent()
        # what_move = row.data(0,0)
        # a = prev_parent.data(0,0)
        # print(a)
        return QTreeWidget.startDrag(self, actions)
    def dropEvent(self, event):
        # widgetItemThatMoved = event.source().currentItem()
        # parentThatReceivedIt = self.itemAt(event.pos())
        # my_custom_collback(self._prev_parent, self._what_move, self._new_parent)  # <- that what i needed
        return QTreeWidget.dropEvent(self, event)


        # print(e.dropAction(), 'baseact', Qt.CopyAction)
        # # if e.keyboardModifiers() & QtCore.Qt.AltModifier:
        # #     #e.setDropAction(QtCore.Qt.CopyAction)
        # #     print('copy')
        # # else:
        # #     #e.setDropAction(QtCore.Qt.MoveAction)
        # #     print("drop")
        #
        # print(e.dropAction())
        # #super(Tree, self).dropEvent(e)
        # index = self.indexAt(e.pos())
        # parent = index.parent()
        # print('in', index.row())
        # self.model().dropMimeData(e.mimeData(), e.dropAction(), index.row(), index.column(), parent)
        #
        # e.accept()

        # def dropEvent(self, event):
        #     if event.mimeData().hasUrls:
        #         event.setDropAction(QtCore.Qt.CopyAction)
        #         event.accept()
        #         # to get a list of files:
        #         drop_list = []
        #         for url in event.mimeData().urls():
        #             drop_list.append(str(url.toLocalFile()))
        #         # handle the list here
        #     else:
        #         event.ignore()
        # def dropEvent(self, event):
        #     """
        #     Event handler for `QDropEvent` events.
        #
        #     If an icon is dropped on a free area of the tree view then the
        #     icon URL is converted to a path (which we assume to be an `HDF5`
        #     file path) and ``ViTables`` tries to open it.
        #
        #     :Parameter event: the event being processed.
        #     """
        #
        #     mime_data = event.mimeData()
        #     if mime_data.hasFormat('text/uri-list'):
        #         if self.dbt_model.dropMimeData(mime_data, Qt.CopyAction, -1, -1, self.currentIndex()):
        #             event.setDropAction(Qt.CopyAction)
        #             event.accept()
        #             self.setFocus(True)
        #     else:
        #         QTreeView.dropEvent(self, event)
