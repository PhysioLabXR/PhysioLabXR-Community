import inspect
import warnings

from PyQt6.QtGui import QPixmap, QPainter, QFont
from PyQt6.QtWidgets import QApplication, QSplashScreen, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal, QCoreApplication

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.version.version import VERSION


class SplashScreenSingleton:
    """
    Singleton class use to decorate LoadingTextNotifier class.
    """
    _instances = {}

    def __call__(self, cls):
        def wrapper(*args, **kwargs):
            if cls not in self._instances:
                self._instances[cls] = cls(*args, **kwargs)
            return self._instances[cls]
        return wrapper

@SplashScreenSingleton()
class SplashLoadingTextNotifier(QObject):
    """
    Anywhere before the MainWindow is shown, and while the splash screen is still visible, you may call
        LoadingTextNotifier().setLoadingText(<loading information>)
    to update the loading text on the splash screen.
    """

    loading_text_changed = pyqtSignal(str)
    splash_active = True

    def __init__(self):
        super().__init__()

    def set_loading_text(self, text, also_print=True):
        if self.splash_active:
            if also_print:
                print(text)
            self.loading_text_changed.emit(text)
        else:
            warnings.warn(f"{self.__class__.__name__}.{inspect.stack()[0].function}: Splash screen is not active. Loading text will not be updated.")
            print("Loading text given is: \n {text}")


class SplashScreen(QSplashScreen):
    """
    The splash screen that is shown while the application is loading. It includes a loading text that can be modified by
    calling LoadingTextNotifier().setLoadingText(<loading information>) any where in the code before the MainWindow is shown
    """
    def __init__(self):
        super().__init__()
        self.setPixmap(QPixmap(AppConfigs()._splash_screen))

        layout = QVBoxLayout()
        self.loading_label = QLabel("Loading...")
        self.loading_label .setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.loading_label)

        version_label = QLabel(f"<i>Physiological Laboratory for Mixed Reality v{VERSION}</i>", self)
        version_label .setAlignment(Qt.AlignmentFlag.AlignRight)
        version_label.setGeometry(180, 120, 320, 20)  # Set the x, y, width, and height values as desired

        self.setLayout(layout)
        self.setWindowTitle("Splash Screen")

        SplashLoadingTextNotifier().loading_text_changed.connect(self.updateLoadingText)

    def updateLoadingText(self, text):
        self.loading_label.setText(text)
        QCoreApplication.processEvents()

    def closeEvent(self, event):
        SplashLoadingTextNotifier().splash_active = False
        event.accept()