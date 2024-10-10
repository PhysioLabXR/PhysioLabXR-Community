from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialogButtonBox, QDialog, QVBoxLayout, QLabel, QCheckBox

from physiolabxr.configs.config import settings


def dialog_popup(msg, mode='modal', title='Warning', dialog_name=None, enable_dont_show=False, main_parent=None, buttons=QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel):
    if enable_dont_show:
        try:
            assert dialog_name is not None
        except AssertionError:
            print("dev: to use enable_dont_show, the dialog must have a unique identifier. Add the identifier by giving"
                  "the dialog_name parameter")
            raise AttributeError
        if settings.contains('show_' + dialog_name) and settings.value('show_' + dialog_name) == 'false':
            print('Skipping showing dialog ' + dialog_name)
            return
    dlg = CustomDialog(title, msg, dialog_name, enable_dont_show, buttons=buttons)  # If you pass self, the dialog will be centered over the main window as before.
    if main_parent:
        main_parent.current_dialog = dlg
    if mode=='modal':
        dlg.activateWindow()
        if dlg.exec():
            print("Dialog popup")
        else:
            print("Dialog closed")
    elif mode=='modeless':
        print("Showing modeless dialog")
        dlg.show()
        dlg.activateWindow()
    else:
        raise NotImplementedError
    return dlg


class CustomDialog(QDialog):
    def __init__(self, title, msg, dialog_name, enable_dont_show, parent=None, buttons=QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel):
        super().__init__(parent=parent)

        self.setWindowTitle(title)
        self.dialog_name = dialog_name

        QBtn = buttons

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.msg = msg

        self.layout = QVBoxLayout()
        message = QLabel(str(msg))

        # center message and button
        self.layout.addWidget(message, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.buttonBox, alignment=Qt.AlignmentFlag.AlignCenter)

        if enable_dont_show:
            # self.dont_show_button = QPushButton()
            self.dont_show_button = QCheckBox("Don't show this again")
            self.layout.addWidget(self.dont_show_button)
            self.dont_show_button.stateChanged.connect(self.toggle_dont_show)

        self.setLayout(self.layout)

    def toggle_dont_show(self):
        if self.dont_show_button.isChecked():
            settings.setValue('show_' + self.dialog_name, False)
            print('will NOT show ' + self.dialog_name)
        else:
            settings.setValue('show_' + self.dialog_name, True)
            print('will show ' + self.dialog_name)
