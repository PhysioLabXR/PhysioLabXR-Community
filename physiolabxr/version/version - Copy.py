import toml

def get_project_version(pyproject_path='pyproject.toml'):
    try:
        with open(pyproject_path, 'r') as file:
            pyproject_data = toml.load(file)
        rtn = pyproject_data['project']['version']
    except FileNotFoundError:
        rtn = '1.1.4'
    return rtn