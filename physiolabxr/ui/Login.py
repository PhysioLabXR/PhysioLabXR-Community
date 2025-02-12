import os
import sys
import webbrowser
import requests
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import auth
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QDialog, QMessageBox
from physiolabxr.configs.configs import AppConfigs


class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setupUi()


    def setupUi(self):
        self.setObjectName("Dialog")
        self.resize(425, 358)

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(50, 10, 191, 20))
        self.label.setObjectName("label")

        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setGeometry(QtCore.QRect(40, 150, 331, 41))
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(self)
        self.pushButton_2.setGeometry(QtCore.QRect(40, 270, 331, 41))
        self.pushButton_2.setObjectName("pushButton_2")

        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setGeometry(QtCore.QRect(40, 70, 331, 21))
        self.lineEdit.setObjectName("lineEdit")

        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(40, 50, 58, 16))
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.setGeometry(QtCore.QRect(40, 100, 58, 16))
        self.label_3.setObjectName("label_3")

        self.lineEdit_2 = QtWidgets.QLineEdit(self)
        self.lineEdit_2.setGeometry(QtCore.QRect(40, 120, 331, 21))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)  # Hide password input

        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setGeometry(QtCore.QRect(40, 240, 251, 16))
        self.label_4.setObjectName("label_4")

        self.checkBox = QtWidgets.QCheckBox(self)
        self.checkBox.setGeometry(QtCore.QRect(80, 200, 251, 20))
        self.checkBox.setObjectName("checkBox")

        self.retranslateUi()
        self.pushButton.clicked.connect(self.handle_login)
        self.pushButton_2.clicked.connect(self.handle_signup)

    def retranslateUi(self):
        self.setWindowTitle("Welcome")
        self.label.setText("Log into your PhysioLabXR account")
        font = QFont()
        font.setBold(True)  # Make the text bold
        font.setPointSize(20)  # Set the font size
        self.label.setFont(font)
        self.label.adjustSize()

        self.pushButton.setText("Log In")
        self.pushButton_2.setText("Sign Up")
        self.label_2.setText("Email:")
        self.label_3.setText("Password:")
        self.label_4.setText("Don't have an account yet?")
        self.checkBox.setText("Remember my account on this device")


    def handle_login(self):
        """Handle user login using Firebase Authentication & Custom Token."""
        email = self.lineEdit.text()
        password = self.lineEdit_2.text()

        if not email or not password:
            QMessageBox.warning(self, "Error", "Please enter both email and password.")
            return

        try:
            # Firebase REST API endpoint for sign-in
            load_dotenv()
            api_key = os.getenv("FIREBASE_API_KEY")
            url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"

            data = {
                "email": email,
                "password": password,
                "returnSecureToken": True,
            }

            response = requests.post(f"{url}?key={api_key}", json=data)

            if response.status_code == 200:
                user_data = response.json()
                uid = user_data.get("localId")

                # Generate a custom token using Firebase Admin SDK
                custom_token = auth.create_custom_token(uid).decode("utf-8")

                if self.checkBox.isChecked():
                    AppConfigs().remembered_uid = uid  # Store UID instead of email

                print("✅ Login Successful - UID stored:", uid)

                self.accept()  # Close the dialog and indicate successful login

            else:
                error_message = response.json().get("error", {}).get("message", "Unknown error occurred.")
                QMessageBox.critical(self, "Login Failed", f"An error occurred: {error_message}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")

    def auto_login(self):
        """Auto-login using stored UID instead of email."""
        uid = AppConfigs().remembered_uid
        print(f"🔍 Checking stored UID in AppConfigs: {uid}")

        if not uid:
            print("❌ No remembered UID found, requiring manual login.")
            return False

        try:
            # Verify user existence using Firebase Admin SDK
            user = auth.get_user(uid)
            if user:
                QMessageBox.information(self, "Auto Login", f"Welcome back, {user.email}!")
                self.accept()
                return True
            else:
                print("❌ UID not found in Firebase, clearing remembered user.")
                AppConfigs().remembered_uid = None
                return False

        except firebase_admin.auth.UserNotFoundError:
            print("❌ User not found in Firebase, clearing remembered user.")
            AppConfigs().remembered_uid = None
            return False
        except Exception as e:
            QMessageBox.critical(self, "Auto Login Failed", f"An unexpected error occurred: {e}")
            AppConfigs().remembered_uid = None
            return False

    def handle_signup(self):
        """Redirect user to the signup webpage."""
        signup_url = "https://storage.googleapis.com/physiolabxr.org/signup.html"
        webbrowser.open(signup_url)
        QMessageBox.information(self, "Redirecting", "You will be redirected to the signup page.")

if __name__ == "__main__":
    # Initialize Firebase Admin SDK if not already initialized
    if not firebase_admin._apps:
        cred = firebase_admin.credentials.Certificate(
            "path/to/serviceAccountKey.json")  # Replace with your Google credentials path
        firebase_admin.initialize_app(cred)

    app = QtWidgets.QApplication(sys.argv)
    dialog = LoginDialog()
    dialog.exec()
