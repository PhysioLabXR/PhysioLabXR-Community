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
  RenaVisualizationTest
  VisualizationLSLChannelTest
  VisualizationZMQChannelTest
  RecordingTest
  ReplayTest
)

# Detect the operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    source venv/bin/activate
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    source venv/bin/activate
elif [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows with Cygwin
    source venv/bin/activate
elif [[ "$OSTYPE" == "msys" ]]; then
    # Windows with MSYS2
    source venv/Scripts/activate
elif [[ "$OSTYPE" == "win32" ]]; then
    # Windows with native shell
    source venv/Scripts/activate
else
    echo "Unknown operating system: $OSTYPE"
    exit 1
fi

export PYTHONPATH="$(pwd)" # add content root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/rena  # add source root to PYTHONPATH
echo "Here is the PYTHONPATH: $PYTHONPATH"

cd rena/tests
# Loop through each test function in the module and run it
for module in "${test_modules[@]}"
do
    echo "Running test: $module"
    pytest $module.py
done

## Wait for the user to press Enter before exiting
read -p "Press Enter to exit"