import os
import json
from typing import List
from fastapi import FastAPI, WebSocket, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# We'll store replay info in memory after loading.
VIDEO_PATH = ""
GAZE_DATA = []
TIMESTAMPS = []

@app.on_event("startup")
def load_data_on_startup():
    """
    Optionally, you can load initial data if you have a known replay file,
    or you can create an endpoint to load it on demand.
    """
    pass  # or load default data

@app.get("/api/gaze_data")
def get_gaze_data():
    """
    Return gaze data for the front-end to overlay or chart.
    Suppose GAZE_DATA is a list of {time, x, y, pupil_size, ...}.
    """
    duration = TIMESTAMPS[-1] if TIMESTAMPS else 0
    return JSONResponse({
        "gaze": GAZE_DATA,
        "duration": duration
    })

@app.get("/api/video_feed")
def video_feed():
    """
    A naive example that returns the entire video file.
    For large files or random seeking, consider partial content range requests.
    """
    if not VIDEO_PATH or not os.path.exists(VIDEO_PATH):
        return Response(content="Video not found", status_code=404)
    return FileResponse(path=VIDEO_PATH, media_type="video/mp4")

@app.websocket("/ws/controls")
async def replay_controls_ws(websocket: WebSocket):
    """
    Optional: A WebSocket for more interactive controls.
    E.g., if the user drags the timeline in the front-end, you can
    send commands to the backend or push real-time updates.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("cmd", "")
            if command == "seek":
                new_time = data.get("time", 0)
                # handle seeking logic, if needed
                # ...
                # respond or broadcast updates
                await websocket.send_json({"status": "OK", "sought_to": new_time})
            elif command == "pause":
                # handle pause
                await websocket.send_json({"status": "paused"})
            elif command == "play":
                # handle play
                await websocket.send_json({"status": "playing"})
    except:
        pass  # on disconnect, just exit

@app.post("/api/load_replay")
def load_replay(file_path: str):
    """
    Example endpoint to load data from a given replay file on demand,
    just like your existing PhysioLabXR logic. You can unify or reuse code
    from your existing replay pipeline.
    """
    global VIDEO_PATH, GAZE_DATA, TIMESTAMPS
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File does not exist"}, status_code=400)

    # Example: assume the folder has "world.mp4" and "eyetrack.json"
    dir_path = os.path.dirname(file_path)
    VIDEO_PATH = os.path.join(dir_path, "world.mp4")
    eyetrack_path = os.path.join(dir_path, "eyetrack.json")

    if not os.path.exists(eyetrack_path):
        return JSONResponse({"error": "No eyetrack.json in directory"}, status_code=400)

    with open(eyetrack_path, "r") as f:
        data = json.load(f)
        times = data.get("times", [])
        xs = data.get("xs", [])
        ys = data.get("ys", [])
        ps = data.get("pupil", [])

        TIMESTAMPS = times
        GAZE_DATA = [
            {"time": t, "x": x, "y": y, "pupil_size": p}
            for t, x, y, p in zip(times, xs, ys, ps)
        ]

    return {"status": "ok", "video_found": os.path.exists(VIDEO_PATH)}

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

def start_backend(host="127.0.0.1", port=8000):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_backend()
