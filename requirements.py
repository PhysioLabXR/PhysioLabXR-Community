import platform
import subprocess
import argparse
import sys


def install_packages(packages):
    """Install packages using pip."""
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")


def main(requirements_path):
    # Define packages to exclude for each OS
    exclude_packages = {
        'windows': [],
        'linux': [],
        'darwin': ['pybluez-updated'],  # darwin is macOS
    }

    # Detect the OS
    os_name = platform.system().lower()

    # Determine which packages to exclude based on the OS
    packages_to_exclude = exclude_packages.get(os_name, [])

    # Initialize list for packages to install
    packages_to_install = []

    # Read requirements.txt and filter based on the exclusion list
    with open(requirements_path, 'r') as req_file:
        for line in req_file:
            package = line.strip()
            # Check if the package is not in the exclusion list
            if package not in packages_to_exclude:
                packages_to_install.append(package)

    # Install the filtered packages
    install_packages(packages_to_install)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Install packages from a requirements file with optional exclusions based on the operating system.")
    parser.add_argument("requirements_path", nargs='?', default="requirements.txt",
                        help="Path to the requirements.txt file (default: requirements.txt)")

    # Parse arguments
    args = parser.parse_args()


    # Call main function with the path to the requirements.txt file
    main(args.requirements_path)