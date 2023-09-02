import os
from typing import List, Union


def get_json_file_paths_from_dir(dir_path: str) -> List[str]:
    files_paths = []
    preset_file_paths = [os.path.join(dir_path, x) for x in dir_path]
    for pf_path in preset_file_paths:
        files_paths.append(pf_path)
    return files_paths

def get_json_file_paths_from_multiple_dir(dir_paths: List[str], flatten=False) -> Union[list[list[str]], list[str]]:
    files_paths = []
    for dir_path in dir_paths:
        if flatten:
            files_paths += get_json_file_paths_from_dir(dir_path)
        else:
            files_paths.append(get_json_file_paths_from_dir(dir_path))
    return files_paths


def get_file_changes(dir_path, last_mod_times):
    files = os.listdir(dir_path)
    current_mod_times = {}
    modifed_files = []
    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.json'):
            current_mod_times[file_name] = os.path.getmtime(file_path)
            if file_name not in last_mod_times or last_mod_times[file_name] != current_mod_times[file_name]:
                print('file {} has been modified'.format(file_name))
                modifed_files.append(file_path)
    return modifed_files, current_mod_times

def get_file_changes_multiple_dir(dir_paths, last_mod_times, flatten=False):
    modified_files = []
    current_mod_times = {}
    for dir_path in dir_paths:
        _modified_files, _current_mod_times = get_file_changes(dir_path, last_mod_times)
        if flatten:
            modified_files += _modified_files
        else:
            modified_files.append(_modified_files)
        current_mod_times.update(_current_mod_times)
    return modified_files, current_mod_times

