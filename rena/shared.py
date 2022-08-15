# replay
FAIL_INFO = 'fail!'
START_COMMAND = 'start!'
START_SUCCESS_INFO = 'start'
VIRTUAL_CLOCK_REQUEST = 'v'

STOP_COMMAND = 'stop!'
STOP_SUCCESS_INFO = 'stop'

# scripting
SCRIPT_STDOUT_MSG_PREFIX = 'S!'
SCRIPT_STOP_REQUEST = 'stop'
SCRIPT_STOP_SUCCESS = 'stopsuccess'
SCRIPT_INFO_REQUEST = 'i'
DATA_BUFFER_PREFIX = 'd'.encode('utf-8')

# rena script
rena_base_script = open("scripting/BaseRenaScript.py", "r").read()

