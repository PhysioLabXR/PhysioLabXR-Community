import os
import time
import platform
import subprocess
import threading
import json
import atexit
import psutil
from pathlib import Path
import numpy as np
from pylsl import resolve_byprop, StreamInlet
from typing import Union, List, Dict, Tuple

from physiolabxr.exceptions.exceptions import (
    FailToSetupDevice,
    CustomDeviceStartStreamError,
    CustomDeviceStreamInterruptedError,
    CustomDeviceNotFoundError,
)


from physiolabxr.interfaces.LSLInletInterface import LSLInletInterface


class BAlert_Interface(LSLInletInterface):
    STATUS_STARTING = 0
    STATUS_INITIALIZING = 1
    STATUS_DETECTING = 2
    STATUS_MEASURING_IMPEDANCE = 3
    STATUS_READY = 4
    STATUS_STREAMING = 5
    STATUS_ERROR = 6
    STATUS_STOPPING = 7
    STATUS_STOPPED = 8

    def __init__(
        self,
        lsl_stream_name: str = "BAlert",
        num_chan: int = 24,
        nominal_srate: float = 256.0,
        exe_path: str | None = None,
        work_dir: str | None = None,
    ):
        super().__init__(lsl_stream_name=lsl_stream_name, num_chan=num_chan)

        if platform.system() != "Windows":
            raise EnvironmentError("Platform Not Supported: BAlert is only supported on Windows")

        self.device_nominal_sampling_rate = nominal_srate
        self.stream_name = lsl_stream_name
        self.stream_type = "EEG"
        self.inlet: StreamInlet | None = None
        self._inlet_closed = False
        self.device_available = False

        self.proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()

        self.exe_path = exe_path or self._guess_exe_path()
        self.work_dir = work_dir or self._guess_work_dir()

        self.status_file = os.path.join(self.work_dir, "Config", "balert_status.txt")
        self.cmd_file = os.path.join(self.work_dir, "Config", "balert_command.txt")

        self._status_watcher = None
        self._status_watcher_stop = threading.Event()

        self._stopping = False

    # =================== Public API ===================
    def _watch_status_loop(self):
        last = None
        seen_down_at = None

        while not self._status_watcher_stop.is_set():
            line = ""
            try:
                with open(self.status_file, "r", encoding="utf-8") as f:
                    line = f.read().strip()
            except Exception:
                pass

            if line and line != last:
                last = line
                parts = line.split(",", 1)
                try:
                    code = int(parts[0])
                except Exception:
                    code = None
                msg = parts[1] if len(parts) > 1 else ""
                base_msg = f"[BAlert] status: {code}, {msg}"

                if code == self.STATUS_ERROR and "Disconnected" in msg:
                    print(base_msg)
                    if seen_down_at is None:
                        seen_down_at = time.time()
                    continue

                if code == self.STATUS_STOPPING:
                    print(base_msg)
                    if seen_down_at is None:
                        seen_down_at = time.time()
                    continue

                if code == self.STATUS_STOPPED:
                    print(base_msg)
                    try:
                        self._stop_reader_thread()
                    except Exception:
                        pass
                    self.proc = None
                    self.device_available = False
                    break

                print(base_msg)

            if seen_down_at and (time.time() - seen_down_at > 3.0):
                try:
                    if self.proc and self.proc.poll() is None:
                        self.proc.terminate()
                        try:
                            self.proc.wait(timeout=1.5)
                        except subprocess.TimeoutExpired:
                            self.proc.kill()
                except Exception:
                    pass
                break

            time.sleep(0.3)

    def _start_status_watcher(self):
        self._status_watcher_stop.clear()
        self._status_watcher = threading.Thread(target=self._watch_status_loop, daemon=True)
        self._status_watcher.start()

    def _stop_status_watcher(self) -> None:
        try:
            self._status_watcher_stop.set()
        except Exception:
            pass

    @staticmethod
    def _sync_license_to_exe_dir(exe_path: Path, user_lic_dir: Path) -> Path:
        exts = {".key", ".exe"}
        if user_lic_dir.is_file():
            user_lic_dir = user_lic_dir.parent
        dst = exe_path.parent / "License"
        dst.mkdir(exist_ok=True)

        for f in list(dst.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                try:
                    f.unlink()
                except Exception:
                    pass

        for f in user_lic_dir.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                (dst / f.name).write_bytes(f.read_bytes())

        return dst

    def start_stream(self, license_dir: str | None = None, exe_path: str | None = None, **kwargs):
        if self.proc and self.proc.poll() is None:
            print("BAlert already running; skip start.")
            return
        if exe_path:
            self.exe_path = exe_path

        if not os.path.isfile(self.exe_path):
            raise FileNotFoundError(f"BAlert.exe not found: {self.exe_path}")

        self.work_dir = self._guess_work_dir()
        self.status_file = os.path.join(self.work_dir, "Config", "balert_status.txt")
        self.cmd_file = os.path.join(self.work_dir, "Config", "balert_command.txt")

        cfg_xml = Path(self.work_dir) / "Config" / "AthenaSDK.xml"
        if not cfg_xml.is_file():
            raise FileNotFoundError(
                f"AthenaSDK.xml not found under work dir: {self.work_dir}\nExpected {cfg_xml}"
            )

        env = os.environ.copy()

        exe_lic = Path(self.exe_path).parent / "License"
        if license_dir:
            lic_dst = self._sync_license_to_exe_dir(Path(self.exe_path), Path(license_dir))
            env["ABM_LICENSE_DIR"] = str(lic_dst)
        else:
            has_any = exe_lic.is_dir() and any(exe_lic.iterdir())
            if not has_any:
                raise FileNotFoundError("License folder not set. Please pick your ABM license folder in the UI.")
            env["ABM_LICENSE_DIR"] = str(exe_lic)

        try:
            if os.path.exists(self.cmd_file):
                open(self.cmd_file, "w", encoding="utf-8").close()
        except Exception:
            pass

        try:
            killed_any = False
            for p in psutil.process_iter(["name"]):
                if (p.info["name"] or "").lower() == "balert.exe":
                    p.kill()
                    killed_any = True
            if killed_any:
                time.sleep(1.0)
        except Exception:
            pass

        if self._try_connect_lsl():
            self.device_available = True
            atexit.register(self.stop_stream)
            return

        creationflags = 0x08000000  # CREATE_NO_WINDOW
        time.sleep(0.3)
        self.proc = subprocess.Popen(
            [self.exe_path],
            cwd=self.work_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )

        self._start_reader_thread()
        ready = self._wait_until_ready_or_lsl(timeout=90)

        if not ready:
            code, msg = self._read_status()
            msg = (msg or "").strip()

            if (code == self.STATUS_STOPPED) and ("Stopped OK" in msg):
                self._cleanup()
                raise CustomDeviceStartStreamError(
                    "B-Alert ESU is not connected or access was denied by the system. "
                    "Please check the USB connection, power/Bluetooth, drivers, and permissions, then try again."
                )
            elif code == self.STATUS_ERROR:
                self._cleanup()
                raise CustomDeviceStartStreamError(f"Failed to start B-Alert: {msg}")
            else:
                self._cleanup()
                raise CustomDeviceStartStreamError(
                    "B-Alert did not become ready in time or no LSL stream was found. "
                    "Please verify the device connection and try again."
                )

        if not self._try_connect_lsl():
            self._cleanup()
            raise CustomDeviceStartStreamError(
                "The process started, but no B-Alert LSL stream was found. "
                "The device may not be connected, or the SDK has not started streaming."
            )

        atexit.register(self.stop_stream)
        self._start_status_watcher()
        self.device_available = True

    def stop_stream(self, **kwargs):
        if getattr(self, "_stopping", False) and self.proc is None and self.inlet is None:
            return
        self._stopping = True

        self._stop_reader_thread()

        if self.inlet is not None:
            try:
                self.inlet.close_stream()
            except Exception:
                pass
        self._inlet_closed = True
        self.inlet = None
        self.streams = None

        try:
            if self.cmd_file:
                with open(self.cmd_file, "w", encoding="utf-8") as f:
                    f.write("STOP")
        except Exception:
            pass

        self._stop_status_watcher()

        if self.proc and self.proc.poll() is None:
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=2.0)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass

        self.proc = None
        self.device_available = False
        self._stopping = False

    # ---- helper: soft fail without raising ----
    def _cleanup(self):
        """Internal cleanup without raising exceptions"""
        try:
            if self.cmd_file:
                with open(self.cmd_file, "w", encoding="utf-8") as f:
                    f.write("STOP")
        except Exception:
            pass

        try:
            self._stop_reader_thread()
            self._stop_status_watcher()
        except Exception:
            pass

        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        except Exception:
            pass

        self.proc = None
        self.inlet = None
        self._inlet_closed = True
        self.device_available = False

    def _soft_fail(self, user_msg: str):
        """Clean up and raise an exception to notify the UI"""
        print(user_msg)
        self._cleanup()
        raise CustomDeviceStartStreamError(user_msg)

    def get_sampling_rate(self):
        return self.device_nominal_sampling_rate

    def is_device_available(self):
        return self.device_available

    def process_frames(self):
        if self.inlet is None or self._inlet_closed:
            return [], [], []
        try:
            frames, timestamps = super().process_frames()
        except Exception:
            return [], [], []
        return frames, timestamps, []

    # ===== Impedance check =====
    def check_impedance(self, timeout: float = 60.0) -> List[Dict]:

        if self.proc and self.proc.poll() is None:
            raise RuntimeError("Acquisition is running, cannot check impedance.")

        try:
            killed_any = False
            for p in psutil.process_iter(["name"]):
                if (p.info["name"] or "").lower() == "balert.exe":
                    p.kill()
                    killed_any = True
            if killed_any:
                time.sleep(0.5)
        except Exception:
            pass

        exe = Path(self.exe_path)
        work = Path(self.work_dir)
        json_path = work / "Config" / "balert_impedance.json"
        status_path = work / "Config" / "balert_status.txt"

        for pth in (json_path, status_path):
            try:
                if pth.exists():
                    pth.unlink()
            except Exception:
                pass

        env = os.environ.copy()
        exe_lic = exe.parent / "License"
        if exe_lic.is_dir() and any(exe_lic.iterdir()):
            env["ABM_LICENSE_DIR"] = str(exe_lic)
        elif "ABM_LICENSE_DIR" in env:
            pass
        else:
            raise FileNotFoundError("No ABM license found. Please select license in Options first.")

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        Payload = Union[List[Dict], str]  # ok=True -> List[Dict]; ok=False -> str

        def one_attempt() -> Tuple[bool, Payload]:
            for pth in (json_path, status_path):
                try:
                    if pth.exists():
                        pth.unlink()
                except Exception:
                    pass

            proc = subprocess.Popen(
                [str(exe), "--impedance-only"],
                cwd=str(work),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=creationflags,
            )

            t0 = time.time()
            last_log = ""
            sig_line = ""
            sig_keywords = (
                "IsConnectionOpened=FALSE",
                "Disconnected",
                "No device",
                "ACCESS_DENIED",
                "Stopped OK",
            )

            while True:
                rc = proc.poll()

                if proc.stdout and not proc.stdout.closed:
                    try:
                        line = proc.stdout.readline()
                        if line:
                            last_log = line.rstrip()
                            if any(k in last_log for k in sig_keywords):
                                sig_line = last_log
                    except Exception:
                        pass

                if json_path.exists():
                    pass

                try:
                    if status_path.exists():
                        s = status_path.read_text("utf-8").strip()
                        if s:
                            parts = s.split(",", 1)
                            code = int(parts[0])
                            smsg = parts[1] if len(parts) > 1 else ""
                            if code in (6, 8) and ("Stopped OK" in smsg or "Disconnected" in smsg):
                                return False, f"disconnected::{smsg}"
                except Exception:
                    pass

                if rc is not None:
                    if sig_line:
                        return False, f"disconnected::{sig_line}"
                    elapsed = time.time() - t0
                    return False, f"early_exit::{elapsed:.2f}::{(last_log or '')}"

                if time.time() - t0 > timeout:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    return False, f"Timeout. Last log: {last_log or 'none'}"

                time.sleep(0.08)

        ok, payload = one_attempt()
        if ok:
            return payload  # type: ignore[return-value]

        if isinstance(payload, str) and payload.startswith("early_exit::"):
            try:
                elapsed = float(payload.split("::", 2)[1])
            except Exception:
                elapsed = 0.0
            if elapsed < 1.8:
                time.sleep(0.6)
                ok2, payload2 = one_attempt()
                if ok2:
                    return payload2  # type: ignore[return-value]
                payload = payload2

            last = payload.split("::", 2)[-1] if isinstance(payload, str) else ""
            raise RuntimeError(
                "B-Alert exited before producing impedance results."
            )

        raise RuntimeError(f"Impedance failed: {payload}")

    # =================== Internals ===================
    def _guess_exe_path(self) -> str:
        here = Path(__file__).resolve().parent
        candidates = [
            here / "x64" / "Release" / "BAlert.exe",
        ]
        for p in candidates:
            if p.is_file():
                return str(p)
        raise FileNotFoundError(
            "BAlert.exe not found in:\n" + "\n".join(str(p) for p in candidates)
        )

    def _guess_work_dir(self) -> str:
        here = Path(__file__).resolve().parent
        tried = []
        for anc in [here] + list(here.parents):
            if anc.name.lower() == "physiolabxr":
                base = anc / "third_party" / "BAlert"
                cfg = base / "Config" / "AthenaSDK.xml"
                tried.append(str(cfg))
                if cfg.is_file():
                    return str(base)
        raise FileNotFoundError(
            "AthenaSDK.xml not found. Looked for:\n" + "\n".join(tried)
        )

    def _read_status(self):
        try:
            with open(self.status_file, "r", encoding="utf-8") as f:
                line = f.read().strip()
            if not line:
                return None, ""
            parts = line.split(",", 1)
            code = int(parts[0])
            msg = parts[1] if len(parts) > 1 else ""
            return code, msg
        except Exception:
            return None, ""

    def _wait_until_ready_or_lsl(self, timeout=90) -> bool:
        t0 = time.time()
        last_msg = ""
        while time.time() - t0 < timeout:
            if self._has_lsl_stream():
                return True
            code, msg = self._read_status()
            if msg and msg != last_msg:
                last_msg = msg
                print(f"[BAlert] status: {code}, {msg}")
            if code in (self.STATUS_READY, self.STATUS_STREAMING):
                return True
            if code in (self.STATUS_ERROR, self.STATUS_STOPPED):
                return False
            time.sleep(0.2)
        return False

    def _has_lsl_stream(self) -> bool:
        try:
            streams = resolve_byprop("name", self.stream_name, timeout=0.3)
            if streams:
                return True
            eeg_streams = resolve_byprop("type", "EEG", timeout=0.3)
            return any(s.name() == self.stream_name for s in eeg_streams)
        except Exception:
            return False

    def _try_connect_lsl(self) -> bool:
        streams = resolve_byprop("name", self.stream_name, timeout=2.0)
        if not streams:
            eeg_streams = resolve_byprop("type", "EEG", timeout=2.0)
            streams = [s for s in eeg_streams if s.name() == self.stream_name]

        print("LSL streams found:",
              [(s.name(), s.type(), s.channel_count(), s.nominal_srate()) for s in streams])

        if not streams:
            return False

        info = streams[0]
        n_ch = info.channel_count()
        if n_ch != self.lsl_num_chan:
            raise RuntimeError(f"channels mismatch: preset {self.lsl_num_chan} vs stream {n_ch}")

        self.inlet = StreamInlet(info, max_buflen=60, recover=False)
        self._inlet_closed = False
        self.streams = streams
        return True

    # ---------- stdout reader ----------
    def _reader_loop(self):
        if not self.proc or not self.proc.stdout:
            return
        while not self._reader_stop.is_set():
            line = self.proc.stdout.readline()
            if not line:
                break
            try:
                print("[BAlert.exe]", line.rstrip())
            except Exception:
                pass

    def _start_reader_thread(self):
        self._reader_stop.clear()
        if self.proc and self.proc.stdout:
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()

    def _stop_reader_thread(self):
        self._reader_stop.set()
        t = self._reader_thread
        self._reader_thread = None
        if t and t.is_alive():
            try:
                t.join(timeout=1.0)
            except Exception:
                pass


if __name__ == "__main__":
    dev = BAlert_Interface(lsl_stream_name="BAlert", num_chan=24, nominal_srate=256.0)
    lic = os.environ.get("ABM_LICENSE_DIR", None)
    dev.start_stream(license_dir=lic)
    try:
        for _ in range(200):
            frames, timestamps, _ = dev.process_frames()
            if len(timestamps):
                print("shape:", np.array(frames).shape, "n_ts:", len(timestamps))
            time.sleep(0.05)
    finally:
        dev.stop_stream()