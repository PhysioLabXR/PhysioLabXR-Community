# from PyQt5 import QtCore, QtGui
# from PyQt5.QtWidgets import QWidget, QApplication, QPushButton
#
#
# class DragButton(QPushButton):
#
#     def mousePressEvent(self, event):
#         self.__mousePressPos = None
#         self.__mouseMovePos = None
#         if event.button() == QtCore.Qt.LeftButton:
#             self.__mousePressPos = event.globalPos()
#             self.__mouseMovePos = event.globalPos()
#
#         super(DragButton, self).mousePressEvent(event)
#
#     def mouseMoveEvent(self, event):
#         if event.buttons() == QtCore.Qt.LeftButton:
#             # adjust offset from clicked point to origin of widget
#             currPos = self.mapToGlobal(self.pos())
#             globalPos = event.globalPos()
#             diff = globalPos - self.__mouseMovePos
#             newPos = self.mapFromGlobal(currPos + diff)
#             self.move(newPos)
#
#             self.__mouseMovePos = globalPos
#
#         super(DragButton, self).mouseMoveEvent(event)
#
#     def mouseReleaseEvent(self, event):
#         if self.__mousePressPos is not None:
#             moved = event.globalPos() - self.__mousePressPos
#             if moved.manhattanLength() > 3:
#                 event.ignore()
#                 return
#
#         super(DragButton, self).mouseReleaseEvent(event)
#
# def clicked():
#     print ("click as normal!")
#
# if __name__ == "__main__":
#     app = QApplication([])
#     w = QWidget()
#     w.resize(800,600)
#
#     button = DragButton("Drag", w)
#     button.clicked.connect(clicked)
#
#     w.show()
#     app.exec_()


import sys
from PyQt5 import QtCore, QtGui, QtWidgets


class ThumbListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super(ThumbListWidget, self).__init__(parent)
        self.setAcceptDrops(True)
        self.setIconSize(QtCore.QSize(124, 124))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            self.dropped.emit(links)
        else:
            event.ignore()


class Dialog_01(QtWidgets.QMainWindow):
    def __init__(self):
        super(Dialog_01, self).__init__()
        self.listItems = {}

        myQWidget = QtWidgets.QWidget()
        myBoxLayout = QtWidgets.QVBoxLayout()
        myQWidget.setLayout(myBoxLayout)
        self.setCentralWidget(myQWidget)

        self.listWidgetA = ThumbListWidget(self)
        for i in range(12):
            QtWidgets.QListWidgetItem('Item ' + str(i), self.listWidgetA)
        myBoxLayout.addWidget(self.listWidgetA)

        self.listWidgetB = ThumbListWidget(self)
        myBoxLayout.addWidget(self.listWidgetB)

        self.listWidgetA.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.listWidgetA.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listWidgetA.dropped.connect(self.items_dropped)
        self.listWidgetA.currentItemChanged.connect(self.item_clicked)

        self.listWidgetB.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.listWidgetB.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listWidgetB.dropped.connect(self.items_dropped)
        self.listWidgetB.currentItemChanged.connect(self.item_clicked)

    def items_dropped(self, arg):
        print(arg)

    def item_clicked(self, arg):
        print(arg)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    dialog_1 = Dialog_01()
    dialog_1.show()
    dialog_1.resize(480, 320)
    sys.exit(app.exec_())