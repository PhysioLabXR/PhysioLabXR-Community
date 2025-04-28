import os
import json
from typing import List

import numpy as np
import cv2
from fastapi import File, UploadFile, Form
from fastapi import FastAPI, WebSocket, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from physiolabxr.utils.buffers import DataBuffer
from physiolabxr.utils.user_utils import stream_in, stream_in_bytes, FileType

app = FastAPI()

# We'll store replay info in memory after loading.
DATA_BUFFER: DataBuffer = DataBuffer()  # global data buffer holds the data from the replay file
VIDEO_RES = [720, 1280, 3]
VIDEO_STREAM_NAME = 'PupilLabsWorldCamera'
N_VIDEO_FRAMES: int = -1


@app.get("/api/gaze_aligned")
def get_gaze_aligned():
    """
    Return x_aligned, y_aligned arrays of length num_video_frames,
    where each element is the nearest gaze coordinate in actual pixel space.
    """
    global DATA_BUFFER, N_VIDEO_FRAMES

    if "PupilLabsWorldCamera" not in DATA_BUFFER.buffer or "PupilLabsEye" not in DATA_BUFFER.buffer:
        return JSONResponse({"error": "Missing camera or gaze data"}, status_code=400)

    # 1) video data
    (video_array, video_timestamps) = DATA_BUFFER.buffer["PupilLabsWorldCamera"]

    # 2) gaze data
    (xy_data, gaze_timestamps) = DATA_BUFFER.buffer["PupilLabsEye"]
    # xy_data shape = (2, t_gaze)
    # gaze_timestamps shape = (t_gaze,)

    import numpy as np
    from bisect import bisect_left

    def find_closest_idx(sorted_array, value):
        pos = bisect_left(sorted_array, value)
        if pos == 0:
            return 0
        if pos == len(sorted_array):
            return len(sorted_array) - 1
        before = sorted_array[pos - 1]
        after = sorted_array[pos]
        if abs(after - value) < abs(value - before):
            return pos
        else:
            return pos - 1

    video_ts = video_timestamps
    gaze_ts = gaze_timestamps

    x_aligned = np.zeros(N_VIDEO_FRAMES, dtype=np.float32)
    y_aligned = np.zeros(N_VIDEO_FRAMES, dtype=np.float32)

    for i in range(N_VIDEO_FRAMES):
        T = video_ts[i]
        idx_g = find_closest_idx(gaze_ts, T)
        # If xy_data is normalized in [0..1], multiply by (width, height)
        x_aligned[i] = xy_data[0, idx_g]
        y_aligned[i] = xy_data[1, idx_g]

    return {
        "x_aligned": x_aligned.tolist(),
        "y_aligned": y_aligned.tolist()
    }

@app.get("/api/pupil_data")
def pupil_data():
    """
    Return JSON with:
       right_times: [t1, t2, ...] in seconds
       right_sizes: [s1, s2, ...] in mm
       left_times:  [...]
       left_sizes:  [...]
    """
    global DATA_BUFFER
    if (
        "PupilLabsPupilR" not in DATA_BUFFER.buffer
        or "PupilLabsPupilL" not in DATA_BUFFER.buffer
    ):
        return JSONResponse({"error": "Missing pupil data"}, status_code=400)

    size_r, ts_r = DATA_BUFFER.buffer["PupilLabsPupilR"]  # shape (1, t_pupil_r) and (t_pupil_r,)
    size_l, ts_l = DATA_BUFFER.buffer["PupilLabsPupilL"]  # shape (1, t_pupil_l) and (t_pupil_l,)

    # Flatten from shape (1, N) => (N,) if needed
    size_r = size_r[0].tolist()
    size_l = size_l[0].tolist()

    # give the relative time
    ts_r = ts_r - ts_r[0]
    ts_l = ts_l - ts_l[0]

    ts_r = ts_r.tolist()
    ts_l = ts_l.tolist()

    return {
        "right_times": ts_r,
        "right_sizes": size_r,
        "left_times": ts_l,
        "left_sizes": size_l
    }

@app.get("/api/eeg_data")
def get_eeg_data():
    global DATA_BUFFER
    if "EEG" not in DATA_BUFFER.buffer:
        return JSONResponse({"error": "Missing EEG data"}, status_code=400)

    eeg_data, eeg_ts = DATA_BUFFER.buffer["EEG"]  # shape: (n_channels, T), (T,)
    n_channels, n_samples = eeg_data.shape

    # Convert timestamps to relative time
    eeg_ts = eeg_ts - eeg_ts[0]

    # Build a flexible structure for variable number of channels
    # Example channel names: CH1, CH2, ...
    channels = []
    for i in range(n_channels):
        channels.append({
            "name": f"CH{i+1}",
            "values": eeg_data[i, :].tolist()
        })

    return {
        "timestamps": eeg_ts.tolist(),
        "channels": channels
    }


@app.post("/api/upload_replay")
async def upload_replay(replay_file: UploadFile = File(...)):
    """
    Example endpoint to load data from a given replay file on demand,
    just like your existing PhysioLabXR logic. You can unify or reuse code
    from your existing replay pipeline.
    """
    global DATA_BUFFER, N_VIDEO_FRAMES
    print("/api/upload_replay: reading file content")
    file_type = FileType.from_filename(replay_file.filename)
    # assert not a csv
    assert file_type != FileType.CSV, "CSV files are not supported yet"
    contents = await replay_file.read()
    print(f"/api/upload_replay: Received {replay_file.filename} with size {len(contents)} bytes")

    file_data = stream_in_bytes(contents, file_type)

    N_VIDEO_FRAMES = len(file_data['PupilLabsWorldCamera'][1])
    file_data['PupilLabsWorldCamera'][0] = file_data['PupilLabsWorldCamera'][0].reshape(VIDEO_RES + [N_VIDEO_FRAMES])
    DATA_BUFFER.buffer = file_data

    return {"status": "ok", "message": "Replay loaded in memory"}


@app.get("/api/video_info")
def video_info():
    """
    Returns:
      - frame_count: total number of frames
      - timestamps: array of timestamps, so the frontend can display them
    """
    global DATA_BUFFER, N_VIDEO_FRAMES
    if DATA_BUFFER is None or VIDEO_STREAM_NAME not in DATA_BUFFER.buffer:
        return JSONResponse({"error": "No video loaded"}, status_code=400)

    data, timestamps = DATA_BUFFER.buffer[VIDEO_STREAM_NAME]  # data shape = (pixels, frames)

    total_duration_sec = timestamps[-1] - timestamps[0]
    # give the relative time
    timestamps = timestamps - timestamps[0]

    # Convert numpy array to a Python list is big, so let's only return the timestamps as a list
    # (but watch out for performance if there are thousands of frames).
    ts_list = timestamps.tolist()
    average_fps = N_VIDEO_FRAMES / total_duration_sec if total_duration_sec > 0 else 0

    return {
        "frame_count": N_VIDEO_FRAMES,
        "timestamps": ts_list,
        "total_duration_sec": total_duration_sec,
        "average_fps": average_fps,
        "video_resolution": VIDEO_RES,
    }


@app.get("/api/frame/{request_frame_index}")
def get_frame(request_frame_index: int):
    """
    Return the specified frame as a JPEG image, by extracting column 'frame_index'
    from the video array := (1280, 720, 3)
    """
    global DATA_BUFFER
    print("/api/frame/{request_frame_index}: called")
    if DATA_BUFFER is None or VIDEO_STREAM_NAME not in DATA_BUFFER.buffer:
        return JSONResponse({"error": "No video loaded"}, status_code=400)

    data, timestamps = DATA_BUFFER.buffer[VIDEO_STREAM_NAME]
    # data shape = (1280, 720, 3, frames)

    if request_frame_index < 0 or request_frame_index >= N_VIDEO_FRAMES:
        return JSONResponse({"error": "Invalid frame index"}, status_code=400)

    frame = data[..., request_frame_index]

    # 3) OpenCV expects BGR channel order usually, but if your data is actually RGB,
    #    you might need to reorder channels. For now assume it's BGR or you don't mind the color swap.

    # 4) Encode to JPEG in memory
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return JSONResponse({"error": "Failed to encode image"}, status_code=500)

    # 5) Return as image/jpeg
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

# Serve the built frontend (optional):
# If you build your Vue/React app into a 'dist' folder, you can serve it with:
#   app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")
# and then optionally create a route for the index:
@app.get("/")
def serve_frontend():
    """
    If you've built your Vue/React app into a static folder,
    serve the index.html by default so the user sees your UI.
    """
    index_path = os.path.join("frontend/dist", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(content="<h1>Frontend not built</h1>", status_code=200)

def start_backend(host="127.0.0.1", port=8000):  # use port 8000 for local dev
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    print("Hi")
    start_backend()
