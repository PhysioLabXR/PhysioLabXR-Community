export PYTHONPATH="$(pwd)" # add content root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/rena  # add source root to PYTHONPATH
echo "Here is the PYTHONPATH: $PYTHONPATH"

echo "Running dummy test"
cd tests
pytest DummyTest.py