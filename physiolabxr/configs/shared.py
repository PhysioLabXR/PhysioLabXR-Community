# replay

SUCCESS_INFO = 'success!'
FAIL_INFO = 'fail!'
LOAD_COMMAND = 'load!'
LOAD_SUCCESS_INFO = 'load'
VIRTUAL_CLOCK_REQUEST = 'v'

PLAY_PAUSE_COMMAND = 'pp!'
PLAY_PAUSE_SUCCESS_INFO = 'pp'

SLIDER_MOVED_COMMAND = 'sm!'
SLIDER_MOVED_SUCCESS_INFO = 'sm'

STOP_COMMAND = 'stop!'
STOP_SUCCESS_INFO = 'stop'

GO_AHEAD_COMMAND = 'go'
CANCEL_START_REPLAY_COMMAND = 'ds!'

TERMINATE_COMMAND = 't!'
TERMINATE_SUCCESS_COMMAND = 't'

PERFORMANCE_REQUEST_COMMAND = 'p!'

# scripting
SCRIPT_STDOUT_MSG_PREFIX = 'SO!'
SCRIPT_STDERR_MSG_PREFIX = 'SE!'
SCRIPT_STOP_REQUEST = 'stop'
SCRIPT_STOP_SUCCESS = 'stopsuccess'
SCRIPT_INFO_REQUEST = 'i'
DATA_BUFFER_PREFIX = 'd'.encode('utf-8')
SCRIPT_PARAM_CHANGE = 'p'

# try:
#     rena_base_script = open("scripting/BaseRenaScript.py", "r").read()
# except FileNotFoundError:
#     try:
#         rena_base_script = open("../scripting/BaseRenaScript.py", "r").read()
#     except FileNotFoundError:
#         rena_base_script = open("physiolabxr/scripting/BaseRenaScript.py", "r").read()

default_plot_format = {
        'time_series': {'is_valid': 1, 'display':1},
        'image': {'is_valid': 0,
                  'image_format': 'PixelMap',
                  'width': 0,
                  'height': 0,
                  'channel_format': 'Channel Last',
                  'scaling_factor': 1,
                  },
        'bar_chart': {'is_valid': 1,
                     'display':1,
                     'y_max': -0.1,
                     'y_min': 0.0,
                     }
    }

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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'