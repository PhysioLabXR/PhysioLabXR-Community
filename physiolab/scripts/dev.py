"""Start both local services and stop both on Ctrl-C. Run with python3 scripts/dev.py."""
import signal
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
children = []
try:
    children.append(subprocess.Popen(['uv','run','--frozen','uvicorn','physiolab.api:app','--host','127.0.0.1','--port','8008'], cwd=root/'backend'))
    children.append(subprocess.Popen(['npm','run','dev'], cwd=root/'web'))
    while all(p.poll() is None for p in children):
        import time
        time.sleep(0.5)
finally:
    for p in children:
        if p.poll() is None: p.send_signal(signal.SIGINT)
    for p in children:
        try: p.wait(timeout=5)
        except subprocess.TimeoutExpired: p.kill(); p.wait()
