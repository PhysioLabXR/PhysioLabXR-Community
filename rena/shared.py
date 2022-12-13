# replay
from enum import Enum

FAIL_INFO = 'fail!'
START_COMMAND = 'start!'
START_SUCCESS_INFO = 'start'
VIRTUAL_CLOCK_REQUEST = 'v'

PLAY_PAUSE_COMMAND = 'pp!'
PLAY_PAUSE_SUCCESS_INFO = 'pp'

SLIDER_MOVED_COMMAND = 'sm!'
SLIDER_MOVED_SUCCESS_INFO = 'sm'

STOP_COMMAND = 'stop!'
STOP_SUCCESS_INFO = 'stop'

TERMINATE_COMMAND = 't!'
TERMINATE_SUCCESS_COMMAND = 't'

# scripting
SCRIPT_STDOUT_MSG_PREFIX = 'S!'
SCRIPT_STOP_REQUEST = 'stop'
SCRIPT_STOP_SUCCESS = 'stopsuccess'
SCRIPT_INFO_REQUEST = 'i'
DATA_BUFFER_PREFIX = 'd'.encode('utf-8')
SCRIPT_PARAM_CHANGE = 'p'

rena_base_script = open("scripting/BaseRenaScript.py", "r").read()

class ParamChange(Enum):
    ADD = 'a'
    REMOVE = 'r'
    CHANGE = 'c'

def parse_slider_moved_command(command_to_parse):
    """
    SLIDER_MOVED_COMMAND will be in the following format when received from ReplayServer: sm!:{position}
    This function separates the original command string from the position value.

    Params:
        command_to_parse: command received from ReplayServer.

    Returns:
        updated_position: the updated slider position.
    """
    return command_to_parse.split(':')[1]
