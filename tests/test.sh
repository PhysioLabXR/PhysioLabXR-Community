#!/bin/bash
#
# Description: This script must be run from the root directory of the project.
#
# Dependencies: This script requires some dependency.
#
# Usage: ./myscript.sh arg1 arg2
#
# Parameters:
#
#
# Author: Rena team
# Date: 3/31/2023
# Version: 1.0


# Get the name of the Python unittest module from the first argument
test_modules=(

   SetupTest
  Dtest
#  RenaVisualizationTest
#  VisualizationLSLChannelTest
#  VisualizationZMQChannelTest
#  RecordingTest
#  ReplayTest
#  XdfTest
#  CsvTest
#  MatTest
#  RenaScriptingTest
)

warning_text="You should create a venv-dev and install packages using pip install -r requirements-dev.txt"

if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* || "$OSTYPE" == "cygwin" ]]; then
    # Linux, Mac OSX, or Windows with Cygwin
    source venv.dev/Scripts/activate || source venv-dev/bin/activate || source venv/bin/activate
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows with MSYS2 or Windows with native shell
    source venv.dev/Scripts/activate || source venv-dev/Scripts/activate || source venv/Scripts/activate
else
    echo "Unknown operating system: $OSTYPE"
    exit 1
fi

if [ ! -d "venv-dev" ]; then
    echo "WARNING: $warning_text"
fi



export PYTHONPATH="$(pwd)" # add content root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/rena  # add source root to PYTHONPATH
echo "Here is the PYTHONPATH: $PYTHONPATH"

cd tests
# Loop through each test function in the module and run it
for module in "${test_modules[@]}"
do
    echo "Running test: $module"
    pytest -sv $module.py -s
done

## Wait for the user to press Enter before exiting
read -p "Press Enter to exit"
