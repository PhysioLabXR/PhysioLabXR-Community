def physiolabxr():
    import multiprocessing
    import sys
    import webbrowser

    from PyQt6 import QtWidgets
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QSystemTrayIcon, QMenu, QInputDialog, QMessageBox

    from physiolabxr.configs.configs import AppConfigs
    from physiolabxr.configs.NetworkManager import NetworkManager
    from physiolabxr.ui.SplashScreen import SplashScreen
    from physiolabxr.ui.SplashScreen import SplashLoadingTextNotifier
    from physiolabxr.ui.Login import LoginDialog
    import firebase_admin

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

    # Initialize Firebase Admin SDK
    SplashLoadingTextNotifier().set_loading_text("Logging in...")
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    # Initialize the LoginDialog
    login_dialog = LoginDialog()

    # Check if the user successfully logs in
    if login_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
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
    else:
        # Exit if login is unsuccessful
        splash.close()
        QMessageBox.critical(None, "Access Denied", "Login required to access the application.")
        sys.exit()