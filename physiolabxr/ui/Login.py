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


FIREBASE_API_KEY = "AIzaSyD7CJXqoCPtv2GzMQpGLKwTo4MacPqjqnw"

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
        """Handle user login using Firebase Authentication & Store Refresh Token."""
        email = self.lineEdit.text()
        password = self.lineEdit_2.text()

        if not email or not password:
            QMessageBox.warning(self, "Error", "Please enter both email and password.")
            return

        try:
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
            data = {"email": email, "password": password, "returnSecureToken": True}
            response = requests.post(url, json=data)
            response_data = response.json()

            if "idToken" in response_data:
                id_token = response_data["idToken"]
                refresh_token = response_data["refreshToken"]
                local_id = response_data["localId"]

                if self.checkBox.isChecked():
                    AppConfigs().remembered_token = id_token
                    AppConfigs().refresh_token = refresh_token

                print(f"✅ Login Successful - User: {local_id}")

                self.accept()  # Close the dialog on success

            else:
                error_message = response_data.get("error", {}).get("message", "Unknown error occurred.")
                QMessageBox.critical(self, "Login Failed", f"An error occurred: {error_message}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")

    def refresh_id_token(self):
        """Refresh Firebase ID Token using the refresh token."""
        refresh_token = AppConfigs().refresh_token

        if not refresh_token:
            print("❌ No refresh token found, user needs to log in again.")
            return None

        try:
            url = f"https://securetoken.googleapis.com/v1/token?key={FIREBASE_API_KEY}"
            data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
            response = requests.post(url, json=data)
            response_data = response.json()

            if "id_token" in response_data:
                new_id_token = response_data["id_token"]
                new_refresh_token = response_data["refresh_token"]

                # ✅ Update stored tokens
                AppConfigs().remembered_token = new_id_token
                AppConfigs().refresh_token = new_refresh_token

                print("🔄 Token refreshed successfully")
                return new_id_token

            else:
                print("❌ Failed to refresh token")
                return None

        except Exception as e:
            QMessageBox.critical(self, "Token Refresh Failed", f"An unexpected error occurred: {e}")
            return None

    def auto_login(self):
        """Auto-login using stored and refreshed token."""
        id_token = self.refresh_id_token() or AppConfigs().remembered_token

        if not id_token:
            print("❌ No valid token found, requiring manual login.")
            return False

        try:
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={FIREBASE_API_KEY}"
            data = {"idToken": id_token}
            response = requests.post(url, json=data)

            if response.status_code == 200:
                print("✅ Auto-login successful")
                self.accept()
                return True
            else:
                print("❌ Invalid token, requiring manual login.")
                AppConfigs().remembered_token = None
                return False

        except Exception as e:
            QMessageBox.critical(self, "Auto Login Failed", f"An unexpected error occurred: {e}")
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
