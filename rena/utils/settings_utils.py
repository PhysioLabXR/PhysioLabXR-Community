from rena import config

def get_presets():
    lsl_preset_names = list(config.settings.childGroups())
    config.settings.endGroup()
    config.settings.beginGroup('presets/devicepresets')
    device_preset_names = list(config.settings.childGroups())
    config.settings.endGroup()
    config.settings.beginGroup('presets/experimentpresets')
    experiment_preset_names = list(config.settings.childGroups())
    config.settings.endGroup()

def export_preset_to_settings(preset, setting_category):
    assert setting_category == 'lslpresets' or setting_category == 'devicepresets' or setting_category == 'experimentpresets'
    if setting_category == 'experimentpresets':
        config.settings.setValue('presets/experimentpresets/{0}/PresetStreamNames'.format(preset[0]), preset[1])
    else:
        config.settings.beginGroup('presets/{0}'.format(setting_category))

        for preset_key, value in preset.items():
            if preset_key != 'GroupChannelsInPlot':
                config.settings.setValue('{0}/{1}'.format(preset['StreamName'], preset_key), value)

        for group_name, group_info_dict in preset['GroupChannelsInPlot'].items():
            for group_info_key, group_info_value in group_info_dict.items():
                config.settings.setValue('{0}/GroupChannelsInPlot/{1}'.format(preset['StreamName'], group_info_key), group_info_value)
        config.settings.endGroup()