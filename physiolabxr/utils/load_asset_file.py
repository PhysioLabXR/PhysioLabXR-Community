import os

from PyQt6.QtGui import QIcon


def load_media_icon(file_name):
    if os.path.exists(f'../media/icons/{file_name}'):
        icon = QIcon(f'../media/icons/{file_name}')
    elif os.path.exists(f'media/icons/{file_name}'):
        icon = QIcon(f'media/icons/{file_name}')
    elif os.path.exists(f'../../media/icons/{file_name}'):
        icon = QIcon(f'../../media/icons/{file_name}')
    else:
        return None
    return icon