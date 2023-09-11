import faulthandler

import pytest
from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton


# Import your application module here
# from your_app_module import YourMainWindow

# A dummy PyQt application for testing
class DummyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.button = QPushButton("Click me", self)
        self.button.clicked.connect(self.on_button_click)

    def on_button_click(self):
        self.button.setText("Clicked!")

@pytest.fixture
def app(qtbot):
    # Create a QApplication instance for testing
    test_app = QApplication([])

    # Create a dummy application window and show it
    window = DummyApp()
    window.show()

    # QtBot is used to simulate user actions
    qtbot.addWidget(window)

    yield window

    # Clean up after the test
    window.close()

def test_button_click(app, qtbot):
    faulthandler.disable()
    # Simulate a button click
    qtbot.mouseClick(app.button, QtCore.Qt.MouseButton.LeftButton)

    # Assert that the button text changes as expected
    assert app.button.text() == "Clicked!"