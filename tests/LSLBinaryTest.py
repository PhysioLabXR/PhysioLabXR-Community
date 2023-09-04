import os.path

import pytest
from physiolabxr.utils.setup_utils import get_lsl_binary


def test_download_lsl_binary() -> None:
    pylsl_lib_path = get_lsl_binary()
    assert os.path.exists(pylsl_lib_path)
    import pylsl  # import pylsl to check if the lib exist