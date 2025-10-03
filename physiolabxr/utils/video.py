import os

import cv2


def _guess_mp4_fourcc():
    """
    Try a few common fourcc codes until one actually opens a VideoWriter.
    Falls back to 'mp4v' which ships with OpenCV on all platforms.
    """
    trial_codes = ("avc1", "H264", "mp4v")
    tmp = "_fourcc_test.mp4"
    for code in trial_codes:
        fourcc = cv2.VideoWriter_fourcc(*code)
        vw = cv2.VideoWriter(tmp, fourcc, 1, (1, 1))
        ok = vw.isOpened()
        vw.release()
        if ok:
            if os.path.exists(tmp):
                os.remove(tmp)
            return fourcc
    return cv2.VideoWriter_fourcc(*"mp4v")

_MP4_FOURCC = _guess_mp4_fourcc()
