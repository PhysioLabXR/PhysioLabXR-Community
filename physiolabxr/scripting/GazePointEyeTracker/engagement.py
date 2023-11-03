import threading
from physiolabxr.scripting.GazePointEyeTracker.GazePointAPI import data_queue

data_store = []

def continuous_data_reader():
    while True:
        data = data_queue.get()
        print(data)
        data_store.append(data)

def start_data_reader():
    thread = threading.Thread(target=continuous_data_reader)
    thread.daemon = True
    thread.start()

def determine_engagement():
    engagement_level = len(data_store)  # Placeholder
    data_store.clear()
    return engagement_level