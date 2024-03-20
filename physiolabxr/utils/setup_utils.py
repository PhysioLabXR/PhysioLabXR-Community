import os.path
import platform
import re
import urllib.request
import warnings
import shutil
import subprocess

from physiolabxr.configs.shared import temp_rpc_path
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

def install_lsl_binary():
    # try import pylsl check if the lib exist
    try:
        import pylsl
    except RuntimeError:
        # the error is LSL binary library file was not found.
        get_lsl_binary()


def is_package_installed(package_name):
    try:
        subprocess.check_output(["dpkg", "-s", package_name], stderr=subprocess.STDOUT, text=True)
        return True
    except subprocess.CalledProcessError:
        return False

def is_brew_installed():
    if shutil.which('brew') is None:
        return False
    else:
        return True


def install_pyaudio():
    # check if we are on mac
    try:
        import pyaudio
    except ModuleNotFoundError:
        if platform.system() == 'Darwin':
            # check if brew is installed
            if not is_brew_installed():
                from PyQt6.QtWidgets import QDialogButtonBox
                dialog_popup("Tried to brew install portaudio, a dependency of pyaudio, necessary for audio interface."
                             "But Brew is not installed, please install brew first from https://brew.sh/. Then restart the app if you need audio streams.",
                             title="Warning", buttons=QDialogButtonBox.StandardButton.Ok)
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


def locate_grpc_tools():
    result = subprocess.run(["dotnet", "nuget", "locals", "global-packages", "--list"], capture_output=True, text=True)
    nuget_cache_path = result.stdout.strip().split(' ')[-1]
    tools_path = os.path.join(nuget_cache_path, "grpc.tools")
    if not os.path.exists(tools_path):
        print("Grpc.Tools not found in NuGet cache.")
        return None
    else:
        return tools_path

def locate_csharp_plugin():
    """Locate the grpc_csharp_plugin in the NuGet package cache."""
    if (tools_path := locate_grpc_tools()) is None:
        return None
    # Find the highest version of Grpc.Tools
    versions = sorted(os.listdir(tools_path), reverse=True)
    if versions:
        plugin_path = os.path.join(tools_path, versions[0], "tools", "macosx_x64", "grpc_csharp_plugin")
        if os.path.exists(plugin_path):
            return plugin_path
    return None


def setup_grpc_csharp_plugin():
    # based on the os install the proper protobuf compiler
    from PyQt6.QtWidgets import QDialogButtonBox
    from physiolabxr.configs.configs import AppConfigs

    if platform.system() == 'Darwin':
        # we don't automate home brew installation because it requires sudo access
        if not is_brew_installed():
            dialog_popup("Tried to brew install dotnet-sdk, necessary for compile remote procedural calls (RPC) for C# (Unity)."
                         "But Brew is not installed, please install brew first from https://brew.sh/. Then restart the app if you need audio streams."
                         "Once brew installed, run 'brew install dotnet-sdk' in your terminal to install dotnet-sdk. Then restart the app if you need compile RPC for C# (Unity). ",
                         title="Warning", buttons=QDialogButtonBox.StandardButton.Ok)
            AppConfigs().is_csharp_plugin_available = False
            return
        # we don't automate dotnet-sdk installation because it requires sudo access
        if not shutil.which('dotnet'):
            dialog_popup("Please brew install dotnet-sdk using 'brew install dotnet-sdk' in your terminal. Then restart the app if you need compile RPC for C# (Unity). ",)
            AppConfigs().is_csharp_plugin_available = False
            return

        # check if csharp plugin is available
        if locate_csharp_plugin() is None:
            os.mkdir(temp_rpc_path)
            subprocess.run(["dotnet", "new", "console"], cwd=temp_rpc_path)
            subprocess.run(["dotnet", "add", "package", "Grpc.Tools"], cwd=temp_rpc_path)

        if (csharp_plugin_path := locate_csharp_plugin()) is None:
            dialog_popup("Unable to automatically configure Grpc.Tools in dotnet as a nuget package. Grpc.Tools not found in NuGet cache. Please install Grpc.Tools manually.", title="Warning", buttons=QDialogButtonBox.StandardButton.Ok)
            AppConfigs().is_csharp_plugin_available = False
        else:
            from physiolabxr.configs.configs import AppConfigs
            AppConfigs().csharp_plugin_path = csharp_plugin_path

    elif platform.system() == 'Windows':
        AppConfigs().is_csharp_plugin_available = False
        warnings.warn("PhysioRPC is not supported on Windows yet.")

    elif platform.system() == 'Linux':
        AppConfigs().is_csharp_plugin_available = False
        warnings.warn("PhysioRPC is not supported on Linux yet.")


def run_setup_check():
    install_lsl_binary()
    install_pyaudio()
    setup_grpc_csharp_plugin()
