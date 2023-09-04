import os.path
import platform
import urllib.request
import warnings
import shutil


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
        import subprocess
        subprocess.run(["dpkg", "-x", binary_name, output_directory])
    # delete the downloaded compressed file
    os.remove(binary_name)
    return output_directory

def get_lsl_binary():
    import site
    site_packages_path = site.getsitepackages()[0]
    pylsl_path = os.path.join(site_packages_path, 'pylsl')
    output_directory = download_lsl_binary()
    # move the extracted lib folder to the lsl site-package's lib folder
    downloaded_lib_path = os.path.join(output_directory, 'usr', 'lib')
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
    print(f"LSL binary installed successfully to {downloaded_lib_path}")
    return pylsl_lib_path