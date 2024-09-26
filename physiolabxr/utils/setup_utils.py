import os.path
import os.path
import platform
import re
import sys
import urllib.request
import warnings
import shutil
import subprocess
import stat

from physiolabxr.configs.shared import temp_rpc_path
from physiolabxr.exceptions.exceptions import RPCCSharpSetupError
from physiolabxr.ui.dialogs import dialog_popup

def remove_readonly(fn, path, excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
        fn(path)
    except Exception as exc:
        print("Skipped:", path, "because:\n", exc)

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
        if 'lsl.dll' not in os.listdir(lib_path := os.path.join(output_directory, 'lib')):
            shutil.move(os.path.join(output_directory, 'bin', 'lsl.dll'), lib_path)
    downloaded_lib_path = os.path.join(output_directory, 'usr', 'lib') if os.path.exists(
        os.path.join(output_directory, 'usr')) else os.path.join(output_directory, 'lib')
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
                         title="Warning", buttons=QDialogButtonBox.StandardButton.Ok, enable_dont_show=True,
                         dialog_name="Darwin brew not installed: LSL")
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
                             title="Warning", buttons=QDialogButtonBox.StandardButton.Ok, enable_dont_show=True,
                             dialog_name="Darwin brew not installed: pyaudio")
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
                             title="Warning", buttons=QDialogButtonBox.StandardButton.Ok, enable_dont_show=True,
                             dialog_name="portaudio not in linux: pyaudio")
                return
        # pip install pyaudio
        print("pip installing pyaudio ...")
        pip_install = subprocess.run(["pip", "install", "pyaudio"])
        if pip_install.returncode == 0:
            print("PyAudio has been successfully installed.")
        else:
            print("Error installing PyAudio:", pip_install.stderr)


def locate_grpc_tools():
    try:
        result = subprocess.run(["dotnet", "nuget", "locals", "global-packages", "--list"], capture_output=True, text=True)
    except FileNotFoundError:
        print("dotnet package doesn't exist")
        return None
    nuget_cache_path = result.stdout.strip().split(': ')[-1]

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

    # Determine the platform and architecture
    os_name = platform.system().lower()
    suffix = '.exe' if os_name == 'windows' else ''

    arch = 'x64' if sys.maxsize > 2 ** 32 else 'x86'
    if os_name == 'linux':
        os_name = 'linux_' + arch
    elif os_name == 'darwin':
        os_name = 'macosx_' + arch
    elif os_name == 'windows':
        os_name = 'windows_' + arch

    # Find the highest version of Grpc.Tools
    versions = sorted(os.listdir(tools_path), reverse=True)
    if versions:
        # Attempt to find a plugin for the current platform and architecture
        for version in versions:
            plugin_path = os.path.join(tools_path, version, 'tools', os_name, f'grpc_csharp_plugin{suffix}')
            if os.path.exists(plugin_path):
                return plugin_path
    return None


def add_grpc_plugin_with_dummy_project():
    # create a dummy project
    if locate_csharp_plugin() is None:
        os.makedirs(temp_rpc_path, exist_ok=True)
        subprocess.run(["dotnet", "new", "console", "--force"], cwd=temp_rpc_path)
        subprocess.run(["dotnet", "add", "package", "Grpc.Tools"], cwd=temp_rpc_path)
    else:
        print("Grpc.Tools already installed in NuGet cache.")
    # check if the plugin is available
    if (csharp_plugin_path := locate_csharp_plugin()) is None:
        dialog_popup(
            "When setting up RPC for C#, unable to automatically configure Grpc.Tools in dotnet as a nuget package. "
            "Grpc.Tools not found in NuGet cache. Please install Grpc.Tools manually."
            "You may ignore this if you don't intend to use RPC for C#", title="Warning", enable_dont_show=True,
            dialog_name="locating csharp plugin failed")
        return None
    return csharp_plugin_path


def add_to_path(new_path):
    """Add a new path to the PATH environment variable.

    Please note this function doesn't modify the PATH permanently. It only updates the PATH in the current environment.
    """
    # Get the current PATH
    current_path = os.environ.get('Path', '')

    # Check if the new_path is already in the PATH
    if new_path in current_path.split(';'):
        print(f"{new_path} is already in Path")
    else:
        # update the PATH in the current environment
        os.environ['PATH'] = f"{current_path};{new_path}"


def add_protoc_to_path_windows():
    winget_info = subprocess.run(["winget", "--info"], capture_output=True, text=True).stdout
    winget_info = winget_info.splitlines()
    winget_info = [x for x in winget_info if 'Portable Package Root (User)' in x][0]
    winget_info = re.sub(r'\s+', ' ', winget_info).split(' ')[-1]
    winget_package_path = os.path.expandvars(winget_info)
    try:
        protobuf_package_dirnames = [x for x in os.listdir(winget_package_path) if 'Google.Protobuf' in x]
    except FileNotFoundError:
        raise RPCCSharpSetupError(f"winget package path: {winget_package_path} doesn't exist.")
    if len(protobuf_package_dirnames) == 0:
        raise RPCCSharpSetupError("Google.Protobuf not found in winget package cache.")
    protobuf_package_dirname = protobuf_package_dirnames[0]
    protobuf_package_bin_path = os.path.join(winget_package_path, protobuf_package_dirname, 'bin')
    add_to_path(protobuf_package_bin_path)
    # protobuf_package_include_path = os.path.join(winget_package_path, protobuf_package_dirname, 'include')
    # add_to_path(protobuf_package_include_path)


def setup_grpc_csharp_plugin():
    # based on the os install the proper protobuf compiler
    from PyQt6.QtWidgets import QDialogButtonBox
    from physiolabxr.configs.configs import AppConfigs

    if platform.system() == 'Darwin':
        # we don't automate home brew installation because it requires sudo access
        if not is_brew_installed():
            dialog_popup(
                "Tried to brew install dotnet-sdk, necessary for compile remote procedural calls (RPC) for C# (Unity)."
                "But Brew is not installed, please install brew first from https://brew.sh/. Then restart the app if you need audio streams."
                "Once brew installed, run 'brew install dotnet-sdk' in your terminal to install dotnet-sdk. "
                "Then restart the app/IDE."
                "You may ignore this if you don't intend to use RPC for C# (Unity). ",
                title="Warning", buttons=QDialogButtonBox.StandardButton.Ok, enable_dont_show=True, dialog_name="setup_grpc_csharp_plugin darwin not brew installed")
            AppConfigs().is_csharp_plugin_available = False
            return
        # we don't automate dotnet-sdk installation because it requires sudo access
        if not shutil.which('dotnet'):
            dialog_popup("Please brew install dotnet-sdk using 'brew install dotnet-sdk' in your terminal."
                         " Then restart the app/IDE if you need compile RPC for C# (Unity). "
                         "You may ignore this if you don't intend to use RPC for C# (Unity). ",
                         title="Warning", buttons=QDialogButtonBox.StandardButton.Ok, enable_dont_show=True, dialog_name="setup_grpc_csharp_plugin not dotnet")
            AppConfigs().is_csharp_plugin_available = False
            return

    elif platform.system() == 'Windows':
        sdk_name = "Microsoft.DotNet.SDK.8"
        result = subprocess.run(["winget", "list", sdk_name, "--accept-source-agreements"], capture_output=True,
                                text=True)

        if sdk_name not in result.stdout:
            result = subprocess.run(["winget", "install", "Microsoft.DotNet.SDK.8", "--accept-source-agreements"])

            if result.returncode == 0:  # if the sdk is successfully installed
                print("Microsoft.DotNet.SDK.8 has been successfully installed.")

                if shutil.which('dotnet') is None:
                    dialog_popup(
                        "DotNet.SDK is installed but dotnet command is not found. Please restart the app if you need to compile RPC for C# (Unity)."
                        "You may ignore this if you don't intend to use RPC for C# (Unity). ",
                        title="Info", buttons=QDialogButtonBox.StandardButton.Ok, enable_dont_show=True, dialog_name="setup_grpc_csharp_plugin not dotnet")
                    AppConfigs().is_csharp_plugin_available = False
                    return

            else:  # winget install protobuf failed
                dialog_popup("Unable to install Microsoft.DotNet.SDK.8 using winget. RPC for C# will not be available."
                             "You may ignore this if you don't intend to use RPC for C# (Unity). ",
                             title="Warning", buttons=QDialogButtonBox.StandardButton.Ok, enable_dont_show=True, dialog_name="setup_grpc_csharp_plugin winget install failed")
                AppConfigs().is_csharp_plugin_available = False
                return

        """
        the command winget list protobuf require user to interact with the terminal to agree to the license
        thus it is not suitable for automation

        protobuf_winget_list_result = subprocess.run(["winget", "list", "protobuf"], capture_output=True, text=True)
        # first check if protobuf is installed via winget, it may be installed but not in PATH
        if protobuf_winget_list_result.returncode == 0 and shutil.which('protoc') is not None:
            # case where protobuf is installed and in PATH :)))
            pass
        elif protobuf_winget_list_result.returncode == 0 and shutil.which('protoc') is None:
            # case where protobuf is installed but not in PATH
            # try to locate protoc and add it to path
            add_protoc_to_path()
        """
        try:
            add_protoc_to_path_windows()
            if shutil.which('protoc') is None:  # if adding to path still not working
                dialog_popup("When setting up RPC, protoc is already installed but not in PATH. "
                             "Please add it to PATH manually if you need to compile RPC for C# (Unity)."
                             "Normally, PhysioLabXR will automatically add it to PATH for you. "
                             "If you believe this is an error, please submit an issue on GitHub."
                             "You may ignore this if you don't intend to use RPC for C# (Unity). ",
                             title="Warning", buttons=QDialogButtonBox.StandardButton.Ok,
                             enable_dont_show=True, dialog_name="setup_grpc_csharp_plugin not protoc")
                AppConfigs().is_csharp_plugin_available = False
        except RPCCSharpSetupError as e:
            # protobuf not found in winget package cache, it needs to be installed
            result = subprocess.run(["winget", "install", "protobuf"])

            if result.returncode == 0 or result.returncode == 0x8A15002B:  # if the protobuf is successfully installed
                print("protobuf has been successfully installed.")
                # try to locate protoc and add it to path
                add_protoc_to_path_windows()
                if shutil.which('protoc') is None:  # if adding to path still not working
                    dialog_popup("When setting up RPC, protoc is already installed but not in PATH. "
                                 "Please add it to PATH manually if you need to compile RPC for C# (Unity)."
                                 "Normally, PhysioLabXR will automatically add it to PATH for you. "
                                 "If you believe this is an error, please submit an issue on GitHub."
                                 "You may ignore this if you don't intend to use RPC for C# (Unity). ",
                                 title="Warning", buttons=QDialogButtonBox.StandardButton.Ok, enable_dont_show=True,
                                 dialog_name="setup_grpc_csharp_adding path but protoc not found")
                    AppConfigs().is_csharp_plugin_available = False
                    return
            else:  # winget install protobuf failed
                dialog_popup("When setting up RPC, Unable to install protobuf using winget. "
                             "Please install from https://github.com/protocolbuffers/protobuf/releases/ and add to PATH manually if you need to compile RPC for C# (Unity)."
                             "You may ignore this if you don't intend to use RPC for C# (Unity). ",
                             title="Warning", buttons=QDialogButtonBox.StandardButton.Ok, enable_dont_show=True,
                             dialog_name="setup_grpc_csharp_adding winget install protobuf failed")
                AppConfigs().is_csharp_plugin_available = False
                return

    elif platform.system() == 'Linux':
        AppConfigs().is_csharp_plugin_available = False
        warnings.warn("PhysioRPC is not supported on Linux yet.")

    # end of setting up dotnet-sdk ###############################################################################
    # check if csharp plugin is available
    csharp_plugin_path = add_grpc_plugin_with_dummy_project()
    if csharp_plugin_path is None:
        dialog_popup(
            "Unable to automatically configure Grpc.Tools in dotnet as a nuget package. Grpc.Tools not found in NuGet cache. Please install Grpc.Tools manually."
            "You may ignore this if you don't intend to use RPC for C# (Unity). ",
            title="Warning", buttons=QDialogButtonBox.StandardButton.Ok, enable_dont_show=True,
            dialog_name="setup_grpc_csharp_plugin csharp plugin not found")
        AppConfigs().is_csharp_plugin_available = False
    else:
        from physiolabxr.configs.configs import AppConfigs
        AppConfigs().csharp_plugin_path = csharp_plugin_path


def run_setup_check():
    install_lsl_binary()
    install_pyaudio()
    setup_grpc_csharp_plugin()
    install_pybluez()
