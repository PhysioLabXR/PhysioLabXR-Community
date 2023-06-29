from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal, QCoreApplication


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
class LoadingTextNotifier(QObject):
    """
    Anywhere before the MainWindow is shown, and while the splash screen is still visible, you may call
        LoadingTextNotifier().setLoadingText(<loading information>)
    to update the loading text on the splash screen.
    """

    loading_text_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def setLoadingText(self, text):
        self.loading_text_changed.emit(text)

class SplashScreen(QSplashScreen):
    """
    The splash screen that is shown while the application is loading. It includes a loading text that can be modified by
    calling LoadingTextNotifier().setLoadingText(<loading information>) any where in the code before the MainWindow is shown
    """
    def __init__(self):
        super().__init__()
        self.setPixmap(QPixmap('../media/logo/RenaLabAppDeprecated.png'))

        layout = QVBoxLayout()
        self.loading_label = QLabel("Loading...")
        self.loading_label .setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.loading_label)

        self.setLayout(layout)
        self.setWindowTitle("Splash Screen")

        LoadingTextNotifier().loading_text_changed.connect(self.updateLoadingText)

    def updateLoadingText(self, text):
        self.loading_label.setText(text)
        QCoreApplication.processEvents()
