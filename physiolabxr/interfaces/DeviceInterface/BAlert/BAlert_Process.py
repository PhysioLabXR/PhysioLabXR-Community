from __future__ import annotations
import os
import time
import subprocess
from pathlib import Path
from typing import Optional
from .BAlert_Options import _guess_work_dir_like_interface
from .BAlert_Options import _sync_license_to_exe_dir


def _exe_path() -> Path:
    from .BAlert_Options import _balert_exe_path
    return _balert_exe_path()


class BAlertProcess:
    """Helper to launch/stop BAlert.exe and talk via status/command files."""
    def __init__(self, license_dir: Optional[str] = None,
                 exe_path: Optional[str | Path] = None):
        self.exe: Path = Path(exe_path) if exe_path else _exe_path()

        self.license_dir = license_dir or os.environ.get("ABM_LICENSE_DIR") or ""
        self.proc: subprocess.Popen | None = None

        self.work_dir: Path = _guess_work_dir_like_interface()

        self.cfg_dir: Path = self.work_dir / "Config"
        self.status_file = self.cfg_dir / "balert_status.txt"
        self.command_file = self.cfg_dir / "balert_command.txt"

    # ---------------- lifecycle ----------------

    def start(self, mode: str = "run", capture_output: bool = False) -> None:
        if not self.exe.exists():
            raise FileNotFoundError(f"BAlert.exe not found at {self.exe}")

        env = os.environ.copy()

        exe_lic = self.exe.parent / "License"
        if self.license_dir:
            lic_dst = _sync_license_to_exe_dir(self.exe, Path(self.license_dir))
            env["ABM_LICENSE_DIR"] = str(lic_dst)
        else:
            if exe_lic.exists() and any(exe_lic.iterdir()):
                env["ABM_LICENSE_DIR"] = str(exe_lic)
            else:
                raise FileNotFoundError(
                    "No ABM license found. Please select a license folder in Options."
                )

        try:
            self.command_file.write_text("", encoding="utf-8")
        except Exception:
            pass

        args = [str(self.exe)]
        if mode == "ping":
            args.append("--version")
        elif mode == "impedance":
            args.append("--impedance-only")

        flags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            flags |= subprocess.CREATE_NO_WINDOW

        stdout = subprocess.PIPE if capture_output else None
        self.proc = subprocess.Popen(
            args,
            cwd=str(self.work_dir),
            env=env,
            creationflags=flags,
            stdout=stdout,
            stderr=subprocess.STDOUT if capture_output else None,
            text=bool(capture_output),
            bufsize=1 if capture_output else -1,
        )

    def stop(self, graceful_timeout: float = 10.0) -> None:

        if not self.proc:
            return

        try:
            self.command_file.write_text("STOP", encoding="utf-8")
        except Exception:
            pass

        if self.proc.poll() is None:
            try:
                self.proc.wait(timeout=graceful_timeout)
            except subprocess.TimeoutExpired:
                try:
                    self.proc.kill()
                except Exception:
                    pass

        self.proc = None

    # ---------------- helpers ----------------

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def read_status(self) -> tuple[Optional[int], str]:
        try:
            line = self.status_file.read_text(encoding="utf-8").strip()
            if not line:
                return None, ""
            parts = line.split(",", 1)
            code = int(parts[0])
            msg = parts[1] if len(parts) > 1 else ""
            return code, msg
        except Exception:
            return None, ""

    def wait_until_ready(self, timeout: float = 90.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            code, _ = self.read_status()
            if code in (4, 5):  # STATUS_READY / STATUS_STREAMING
                return
            if code in (6, 8):  # STATUS_ERROR / STATUS_STOPPED
                raise RuntimeError("BAlert reported error/stop")
            time.sleep(0.2)
        raise TimeoutError("BAlert not ready in time")