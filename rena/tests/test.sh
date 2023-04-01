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
test_module="RenaVisualizationTest"
function_prefix="test_"

test_functions=$(python -c "import rena.tests.${test_module} as ${test_module}; print('\n'.join([f for f in dir(${test_module}) if f.startswith('${function_prefix}')]))")

export PYTHONPATH="$(pwd)" # add content root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/rena  # add source root to PYTHONPATH
echo "Here is the PYTHONPATH: $PYTHONPATH"

cd rena/tests

# Loop through each test function in the module and run it
echo "Starting Tests"
for test_function in $test_functions; do
    echo "calling test function $test_module.$test_function"
    python -m unittest $test_module.$test_function
done

## Wait for the user to press Enter before exiting
read -p "Press Enter to exit"