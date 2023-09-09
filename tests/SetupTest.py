import os.path
import platform
import pytest
from physiolabxr.utils.setup_utils import get_lsl_binary


def test_download_lsl_binary() -> None:
    # the error is LSL binary library file was not found.
    pylsl_lib_path = get_lsl_binary()
    if platform.system() != "Darwin":
        assert os.path.exists(pylsl_lib_path)
    import pylsl  # import pylsl to check if the lib exist
    from tests.test_utils import send_data_time
    send_data_time(run_time=3)  # test if data can be sent without erroring out

