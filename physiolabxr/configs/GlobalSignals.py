from PyQt6.QtCore import pyqtSignal, QObject

from physiolabxr.utils.Singleton import Singleton, SingletonQObject


class GlobalSignals(QObject, metaclass=SingletonQObject):
    """
    Contains signals that are used globally across the application. This is a singleton class. It will be created when
    startup is called.
    @attribute stream_presets_entry_changed_signal: signal emitted when a stream preset entry is changed. This is
    fired by SettingsWidget.reload_stream_presets.
    Coupled with:
        AddWidget needs to update add combobox with the new stream preset entries
        ScriptingWidget needs to update the input combobox with the new stream preset entries
    """
    stream_presets_entry_changed_signal = pyqtSignal()
    stream_preset_nominal_srate_changed = pyqtSignal(tuple)