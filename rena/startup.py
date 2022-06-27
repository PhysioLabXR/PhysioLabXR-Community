from rena import config, config_ui


def load_default_settings():
    if not config.settings.contains('theme') or config.settings.value('theme') is None:
        config.settings.setValue('theme', config_ui.default_theme)
    if not config.settings.contains('file_format') or config.settings.value('file_format') is None:
        config.settings.setValue('file_format', config.DEFAULT_FILE_FORMAT)