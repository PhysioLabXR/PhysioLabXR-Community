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
        self.channel_groups_widgets = []
        self.channel_widgets = []

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
            channel_group = QTreeWidgetItem(self)
            channel_group.setText(0, group_name)
            channel_group.setText(1, group_name)
            channel_group.setFlags(channel_group.flags() |
                            Qt.ItemIsTristate | Qt.ItemIsUserCheckable | Qt.ItemIsEditable |
                            Qt.ItemIsDragEnabled)
            channel_group.setData(0, Qt.DisplayRole, 'group')

            for channel_index in channel_groups_information[group_name]['channels']:
                channel = QTreeWidgetItem(channel_group)
                channel.setFlags(channel.flags()
                # | Qt.ItemIsTristate
                | Qt.ItemIsUserCheckable| Qt.ItemIsEditable
                | Qt.ItemIsDragEnabled
                )
                channel.setText(0, self.preset['ChannelNames'][channel_index])
                channel.setCheckState(0, Qt.Unchecked)
                channel.setData(0, Qt.DisplayRole, 'channel')
                self.channel_widgets.append(channel)
            self.channel_groups_widgets.append(channel_group)
    def startDrag(self, actions):
        # row = self.selectedItems()[0]
        # prev_parent = row.parent()
        # what_move = row.data(0,0)
        # a = prev_parent.data(0,0)
        # print(a)
        dragged = self.selectedItems()
        [c.setDisabled(True) for c in self.channel_widgets if c not in dragged]

        is_any_group_dragged = False
        for d in dragged:
            if d.data(0, Qt.DisplayRole) == 'group':
                is_any_group_dragged = True
                break
        if is_any_group_dragged:
            [g.setDisabled(True) for g in self.channel_groups_widgets if g not in dragged]

        return QTreeWidget.startDrag(self, actions)

    def dropEvent(self, event):
        # widgetItemThatMoved = event.source().currentItem()
        # parentThatReceivedIt = self.itemAt(event.pos())
        # my_custom_collback(self._prev_parent, self._what_move, self._new_parent)  # <- that what i needed
        [c.setDisabled(True) for c in self.channel_widgets]
        [g.setDisabled(True) for g in self.channel_groups_widgets]

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
