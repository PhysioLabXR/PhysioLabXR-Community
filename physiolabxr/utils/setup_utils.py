import os.path
import platform
import urllib.request
import warnings
import shutil
import subprocess

from physiolabxr.ui.dialogs import dialog_popup


def get_ubuntu_version():
    try:
        with open("/etc/os-release", "r") as file:
            for line in file:
                if line.startswith("VERSION_CODENAME="):
                    return line.split("=")[1].strip()
    except Exception:
        pass
    return None

def download_lsl_binary():
    support_ubuntu_versions = {'bionic', 'focal', 'jammy'}
    # Get the user's operating system
    user_os = platform.system()
    repo_url = "https://raw.githubusercontent.com/PhysioLabXR/lsl-binaries/master/"
    # Get the Ubuntu version (only for Ubuntu)
    ubuntu_version = ""
    if user_os == "Linux":
        ubuntu_version = get_ubuntu_version()

    # Determine the system architecture (only for Windows)
    architecture = platform.architecture()[0] if user_os == "Windows" else ""

    # Determine the appropriate download URL based on OS, architecture, and Ubuntu version
    if user_os == "Darwin":  # Mac OS
        binary_name = f"liblsl-1.16.2-OSX_amd64.tar.bz2"
    elif user_os == "Windows":
        if architecture == "64bit":  # 64-bit Windows (amd64)
            binary_name = f"liblsl-1.16.2-Win_amd64.zip"
        else:  # 32-bit Windows (i386)
            binary_name = f"liblsl-1.16.2-Win_i386.zip"
    elif user_os == "Linux":
        if ubuntu_version not in support_ubuntu_versions:
            warnings.warn(f"LSL does not support ubuntu version {ubuntu_version}. "
                          f"Supported versions include {support_ubuntu_versions}."
                          f"Using Jammy (2022) binary, LSL may not function properly.")
            ubuntu_version = 'jammy'
        # Ubuntu with supported versions (bionic, focal, jammy)
        binary_name = f"liblsl-1.16.2-bionic_amd64.deb"
    else:
        # Unsupported OS, architecture, or version
        warnings.warn(f"Unsupported OS {user_os} {architecture}. PyLSL will not be available.")
        return

    # Download the binary
    print(f"Downloading binary for {user_os} {architecture} {ubuntu_version}...")
    urllib.request.urlretrieve(f"{repo_url}{binary_name}", binary_name)

    # unpack the binary and move it to lsl site-package's lib folder
    output_directory = 'unpacked_lsl_binary'
    if binary_name.endswith(".tar.bz2"):
        import tarfile
        with tarfile.open(binary_name, "r:bz2") as tar:
            tar.extractall(path=output_directory)
    elif binary_name.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(binary_name, "r") as zip_ref:
            zip_ref.extractall(output_directory)
    elif binary_name.endswith(".deb"):
        subprocess.run(["dpkg", "-x", binary_name, output_directory])
    # delete the downloaded compressed file
    os.remove(binary_name)
    if user_os == "Windows":
        # move dll from bin to lib
        if 'lsl.dll' not in os.listdir(lib_path:=os.path.join(output_directory, 'lib')):
            shutil.move(os.path.join(output_directory, 'bin', 'lsl.dll'), lib_path)
    downloaded_lib_path = os.path.join(output_directory, 'usr', 'lib') if os.path.exists(os.path.join(output_directory, 'usr')) else os.path.join(output_directory, 'lib')
    return output_directory, downloaded_lib_path

def get_lsl_binary():
    # # mac does not need lsl binary
    # if platform.system() == 'Darwin':
    #     print(f"LSL binary is not needed for MacOS")

    import site
    site_packages_path = [x for x in site.getsitepackages() if "site-packages" in x]

    if len(site_packages_path) == 0:
        warnings.warn(f"Cannot find site-packages path. PyLSL will not be available.")
        return
    site_packages_path = site_packages_path[0]
    pylsl_path = os.path.join(site_packages_path, 'pylsl')
    pylsl_lib_path = None
    if platform.system() == "Darwin":
        if shutil.which('brew') is None:
            from PyQt6.QtWidgets import QDialogButtonBox
            dialog_popup("Tried to brew install labstreaminglayer/tap/lsl, necessary for using LSL interface."
                         "But Brew is not installed, please install brew first from https://brew.sh/. Then restart the app if you need to use pylsl."
                         "Unexpected behavior may occur if you continue to use the app without brew.",
                         title="Warning", buttons=QDialogButtonBox.StandardButton.Ok)
            return
        print("Brew installing lsl library ...")
        subprocess.run(["brew", "install", "labstreaminglayer/tap/lsl"])
        env_command = 'export DYLD_LIBRARY_PATH="/opt/homebrew/lib"'
        subprocess.run(env_command, shell=True)
    else:
        output_directory, downloaded_lib_path = download_lsl_binary()
        assert os.path.exists(pylsl_path), 'pylsl package is not installed, please install pylsl first'
        # move the extracted lib folder to the lsl site-package's lib folder
        if os.path.exists(downloaded_lib_path):
            # remove the lib folder if it exists
            if os.path.exists(pylsl_lib_path := os.path.join(pylsl_path, 'lib')):
                shutil.rmtree(pylsl_lib_path)
            shutil.move(downloaded_lib_path, pylsl_path)
            # delete the extracted folder
            shutil.rmtree(output_directory)
        else:
            warnings.warn(f"lib path {downloaded_lib_path} not found in the downloaded binary. "
                          f"PyLSL will not be available.")
            return
        print(f"LSL binary installed successfully to {pylsl_path}")
    return pylsl_lib_path


def get_pybluez_library():
    if platform.system() == 'Windows':
        subprocess.run(["pip", "install", "pybluez-updated"])

        # check if the installation was successful
        try:
            import bluetooth
        except ImportError:
            warnings.warn("Pybluez is required for UnicornHybridBlack. \n"
                          "Follow https://visualstudio.microsoft.com/visual-cpp-build-tools/ "
                          "to install the required build tools and then run 'pip install pybluez-updated' to install pybluez.")
    else:
        print("Unicorn Hybrid Black is not supported on Darwin or Linux. Pybluez will not be installed.")

def install_lsl_binary():
    # try import pylsl check if the lib exist
    try:
        import pylsl
    except RuntimeError:
        # the error is LSL binary library file was not found.
        get_lsl_binary()

def install_pybluez():
    # try import bluetooth check if the lib exist
    # Note: only windows needs pybluez for UnicornHybridBlack
    try:
        import bluetooth
    except ImportError:
        get_pybluez_library()



def is_package_installed(package_name):
    try:
        subprocess.check_output(["dpkg", "-s", package_name], stderr=subprocess.STDOUT, text=True)
        return True
    except subprocess.CalledProcessError:
        return False

def install_pyaudio():
    # check if we are on mac
    try:
        import pyaudio
    except ModuleNotFoundError:
        if platform.system() == 'Darwin':
            # check if brew is installed
            if shutil.which('brew') is None:
                from PyQt6.QtWidgets import QDialogButtonBox
                dialog_popup("Tried to brew install portaudio, a dependency of pyaudio, necessary for audio interface."
                                  "But Brew is not installed, please install brew first from https://brew.sh/. Then restart the app if you need audio streams.", title="Warning", buttons=QDialogButtonBox.StandardButton.Ok)
                from physiolabxr.configs.configs import AppConfigs
                AppConfigs().is_audio_interface_available = False
                return
            # need to brew install portaudio
            print("Brew installing portaudio ...")
            subprocess.run(["brew", "install", "portaudio"])
        elif platform.system() == "Linux":
            # need to apt install portaudio
            if not is_package_installed("portaudio19-dev") or not is_package_installed("python3-dev"):
                from PyQt6.QtWidgets import QDialogButtonBox
                dialog_popup("To use audio streams on Linux, you need to install portaudio, a dependency of pyaudio.\n"
                             "To do so, please run the following two commands in your terminal: \n"
                             "sudo apt-get install portaudio19-dev\n"
                             "sudo apt install python3-dev\n"
                             "Then restart the app if you need audio streams.",
                             title="Warning", buttons=QDialogButtonBox.StandardButton.Ok)
                return
        # pip install pyaudio
        print("pip installing pyaudio ...")
        pip_install = subprocess.run(["pip", "install", "pyaudio"])
        if pip_install.returncode == 0:
            print("PyAudio has been successfully installed.")
        else:
            print("Error installing PyAudio:", pip_install.stderr)


def run_setup_check():
    install_lsl_binary()
    install_pyaudio()
    install_pybluez()
