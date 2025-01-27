import os
import sys
import webbrowser
import requests
from dotenv import load_dotenv
import firebase_admin
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QDialog, QMessageBox


class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setupUi()

    def setupUi(self):
        self.setObjectName("Dialog")
        self.resize(425, 358)

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(130, 10, 191, 20))
        self.label.setObjectName("label")

        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setGeometry(QtCore.QRect(40, 150, 331, 41))
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(self)
        self.pushButton_2.setGeometry(QtCore.QRect(40, 250, 331, 41))
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
        self.label_4.setGeometry(QtCore.QRect(40, 220, 251, 16))
        self.label_4.setObjectName("label_4")

        self.retranslateUi()
        self.pushButton.clicked.connect(self.handle_login)
        self.pushButton_2.clicked.connect(self.handle_signup)

    def retranslateUi(self):
        self.setWindowTitle("Login")
        self.label.setText("Welcome to PhysioLabXR!")
        self.pushButton.setText("Log In")
        self.pushButton_2.setText("Sign Up")
        self.label_2.setText("Email:")
        self.label_3.setText("Password:")
        self.label_4.setText("Don't have an account yet?")

    def handle_login(self):
        """Handle user login using Firebase Authentication."""
        email = self.lineEdit.text()
        password = self.lineEdit_2.text()

        if not email or not password:
            QMessageBox.warning(self, "Error", "Please enter both email and password.")
            return

        try:
            # Firebase REST API endpoint for sign-in
            url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
            load_dotenv()
            api_key = os.getenv("FIREBASE_API_KEY")
            data = {
                "email": email,
                "password": password,
                "returnSecureToken": True,
            }
            response = requests.post(f"{url}?key={api_key}", json=data)

            if response.status_code == 200:
                user_data = response.json()
                QMessageBox.information(self, "Login Successful", f"Welcome back, {user_data['email']}!")
                self.accept()  # Close the dialog and indicate successful login
            else:
                error_message = response.json().get("error", {}).get("message", "Unknown error occurred.")
                QMessageBox.critical(self, "Login Failed", f"An error occurred: {error_message}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")

    def handle_signup(self):
        """Redirect user to the signup webpage."""
        signup_url = "https://storage.googleapis.com/physiolabxr.org/signup.html"
        webbrowser.open(signup_url)
        QMessageBox.information(self, "Redirecting", "You will be redirected to the signup page.")


if __name__ == "__main__":
    # Initialize Firebase Admin SDK if not already initialized
    if not firebase_admin._apps:
        cred = firebase_admin.credentials.Certificate("path/to/serviceAccountKey.json")  # Replace with your Google credentials path
        firebase_admin.initialize_app(cred)

    app = QtWidgets.QApplication(sys.argv)
    dialog = LoginDialog()
    dialog.exec()
