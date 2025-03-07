import os


def physiolabxr():
    import multiprocessing
    import sys
    import webbrowser

    from PyQt6 import QtWidgets
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QSystemTrayIcon, QMenu, QInputDialog, QMessageBox
    from PyQt6.QtCore import QSettings

    from physiolabxr.configs.configs import AppConfigs
    from physiolabxr.configs.NetworkManager import NetworkManager
    from physiolabxr.ui.SplashScreen import SplashScreen
    from physiolabxr.ui.SplashScreen import SplashLoadingTextNotifier
    from physiolabxr.ui.Login import LoginDialog
    AppConfigs(_reset=False)  # create the singleton app configs object
    NetworkManager()

    app = None

    multiprocessing.freeze_support()  # for built exe

    # load the qt application
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    tray_icon = QSystemTrayIcon(QIcon(AppConfigs()._app_logo), parent=app)
    tray_icon.setToolTip('PhysioLabXR')
    tray_icon.show()

    # create the splash screen
    splash = SplashScreen()
    splash.show()

    SplashLoadingTextNotifier().set_loading_text("Logging in...")

    login_dialog = LoginDialog()

    auto_login_success = login_dialog.auto_login()

    if not auto_login_success:
        if login_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            splash.close()
            QMessageBox.critical(None, "Access Denied", "Login required to access the application.")
            sys.exit()

    # load default settings
    from physiolabxr.utils.setup_utils import run_setup_check
    run_setup_check()
    from physiolabxr.startup.startup import load_settings
    load_settings(revert_to_default=False, reload_presets=False)
    # main window init
    print("Creating main window")
    from physiolabxr.ui.MainWindow import MainWindow
    window = MainWindow(app=app)

    window.setWindowIcon(QIcon(AppConfigs()._app_logo))
    # make tray menu
    menu = QMenu()
    exit_action = menu.addAction('Exit')
    exit_action.triggered.connect(window.close)

    print("Closing splash screen, showing main window")
    # splash screen destroy
    splash.close()
    window.show()

    print("Entering exec loop")
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print('App terminate by KeyboardInterrupt')
        sys.exit()