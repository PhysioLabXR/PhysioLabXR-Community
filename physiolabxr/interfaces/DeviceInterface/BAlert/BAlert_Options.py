from __future__ import annotations
import os
import json
import psutil
import time
import subprocess
from pathlib import Path
from typing import List

import numpy as np
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QHeaderView
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

try:
    import mne
except Exception:
    mne = None

from physiolabxr.ui.BaseDeviceOptions import BaseDeviceOptions
from physiolabxr.configs.configs import AppConfigs

# ----------------- constants -----------------
ORDER = [
    "Fp1", "F7", "F8", "T4", "T6", "T5", "T3", "Fp2",
    "O1", "P3", "Pz", "F3", "Fz", "F4", "C4", "P4",
    "POz", "C3", "Cz", "O2"
]
_NAME_TO_MNE = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8", "POz": "Oz"}

WINDOW_MIN_W, WINDOW_MIN_H = 1000, 860
HEAD_R   = 1.05
PLOT_PAD = 0.22
NOSE_APEX_H = 0.16
NOSE_BASE_IN = 0.03
NOSE_HALF   = 0.085
NOSE_LINE_W = 2.0
EAR_A = 0.09
EAR_B = 0.15
EAR_LINE_W = 2.0
MARGIN = 0.85

# ----------------- helpers -----------------
def _kill_existing_balert(timeout: float = 1.5):
    try:
        killed = False
        for p in psutil.process_iter(['name']):
            if (p.info.get('name') or '').lower() == 'balert.exe':
                p.kill()
                killed = True
        if killed:
            time.sleep(timeout)
    except Exception:
        pass

def _sync_license_to_exe_dir(exe: Path, user_lic_dir: Path) -> Path:
    exts = {".lic", ".key", ".dat", ".xml", ".txt"}
    if user_lic_dir.is_file():
        user_lic_dir = user_lic_dir.parent

    dst = exe.parent / "License"
    dst.mkdir(exist_ok=True)

    try:
        if user_lic_dir.resolve() == dst.resolve():
            return dst
    except Exception:
        pass

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

def build_pos_from_mne(order: list[str]) -> dict[str, tuple[float, float]]:
    ch_pos = {}
    if mne is not None:
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            ch_pos = montage.get_positions().get("ch_pos", {}) or {}
        except Exception:
            ch_pos = {}

    names_mne = [_NAME_TO_MNE.get(ch, ch) for ch in order]
    xyz_list, keep_names = [], []
    for raw_name, mne_name in zip(order, names_mne):
        if mne_name in ch_pos:
            v = np.asarray(ch_pos[mne_name], dtype=float)
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v = v / nrm
            xyz_list.append(v)
            keep_names.append(raw_name)

    if not xyz_list:
        import math
        r = HEAD_R * 0.75
        n = max(len(order), 1)
        return {ch: (r * math.cos(2*np.pi*i/n), r * math.sin(2*np.pi*i/n))
                for i, ch in enumerate(order)}

    xyz = np.vstack(xyz_list)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    theta = np.arctan2(y, x)
    colat = np.arccos(np.clip(z, -1.0, 1.0))
    colat_max = np.percentile(colat, 98) or (np.pi/2)
    if colat_max <= 0:
        colat_max = np.pi / 2
    r = (colat / colat_max) * (MARGIN * HEAD_R)
    px = r * np.cos(theta)
    py = r * np.sin(theta)

    out = {name: (float(ix), float(iy)) for name, ix, iy in zip(keep_names, px, py)}

    R_clip = HEAD_R * 0.98
    for k, (ix, iy) in list(out.items()):
        rad = (ix*ix + iy*iy) ** 0.5
        if rad > R_clip and rad > 0:
            s = R_clip / rad
            out[k] = (ix * s, iy * s)

    middle_flat = {"T3", "C3", "Cz", "C4", "T4"}
    last_arc = {"O1", "POz", "O2"}

    MID_UP = 0.06 * HEAD_R
    ys_mid = [out[n][1] for n in middle_flat if n in out]
    if ys_mid:
        y_flat = float(np.mean(ys_mid)) + MID_UP
        for n in middle_flat:
            if n in out:
                x0, _ = out[n]
                out[n] = (x0, y_flat)

    xs_last = [abs(out[n][0]) for n in last_arc if n in out]
    if xs_last:
        x_max = max(xs_last) or 1.0
        y_base = float(np.mean([out[n][1] for n in last_arc if n in out]))
        AMP = 0.08 * HEAD_R
        POWER = 1.5
        for n in last_arc:
            if n in out:
                x0, _ = out[n]
                k = 1.0 - (abs(x0) / x_max) ** POWER
                new_y = y_base + AMP * k
                out[n] = (x0, new_y - 0.1 * HEAD_R)

    return out

POS: dict[str, tuple[float, float]] = build_pos_from_mne(ORDER)

def color_for_value(v: float, ok: bool, th_g: float = 40.0, th_y: float = 80.0):
    if v == 99999:
        return (0.6, 0.6, 0.6)
    if v < th_g:
        return (0.2, 0.8, 0.2)
    if v < th_y:
        return (1.0, 0.85, 0.2)
    return (1.0, 0.3, 0.3)

def _balert_exe_path() -> Path:
    here = Path(__file__).parent
    cands = [here / "x64" / "Release" / "BAlert.exe"]
    for p in cands:
        if p.is_file():
            return p
    return cands[0]

def _guess_work_dir_like_interface() -> Path:
    here = Path(__file__).resolve().parent
    for anc in [here] + list(here.parents):
        if anc.name.lower() == "physiolabxr":
            base = anc / "third_party" / "BAlert"
            if (base / "Config" / "AthenaSDK.xml").is_file():
                return base
    raise FileNotFoundError("AthenaSDK.xml not found under physiolabxr/third_party/BAlert")

# ----------------- worker -----------------
class _ImpWorker(QtCore.QThread):
    finished_items = QtCore.pyqtSignal(list)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, exe: Path, license_dir: str, work_dir: Path,
                 timeout: float = 60.0, parent=None):
        super().__init__(parent)
        self.exe = exe
        self.license_dir = license_dir
        self.work_dir = work_dir
        self.timeout = timeout
        self.status_path = self.work_dir / "Config" / "balert_status.txt"
        self.json_path   = self.work_dir / "Config" / "balert_impedance.json"
        self.before_mtime = self.json_path.stat().st_mtime if self.json_path.exists() else 0.0

    def run(self):
        try:
            if not self.exe.exists():
                self.failed.emit(f"Cannot find BAlert.exe at: {self.exe}")
                return
            if not self.license_dir or not Path(self.license_dir).exists():
                self.failed.emit("Invalid license folder.")
                return

            def one_attempt() -> tuple[bool, str]:
                for p in (self.json_path, self.status_path):
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception:
                        pass

                env = os.environ.copy()
                env["ABM_LICENSE_DIR"] = self.license_dir

                flags = 0
                if hasattr(subprocess, "CREATE_NO_WINDOW"):
                    flags |= subprocess.CREATE_NO_WINDOW

                time.sleep(0.3)

                proc = subprocess.Popen(
                    [str(self.exe), "--impedance-only"],
                    cwd=str(self.work_dir),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=flags,
                )

                t0 = time.time()
                last_log = ""
                disconnected_hit = False
                disconnect_markers = (
                    "IsConnectionOpened=FALSE",
                    "Disconnected",
                    "No device",
                    "Stopped OK",
                    "STATUS_STOPPED",
                )

                while True:
                    rc = proc.poll()

                    if proc.stdout and not proc.stdout.closed:
                        try:
                            line = proc.stdout.readline()
                            if line:
                                line = line.rstrip()
                                last_log = line
                                if any(m in line for m in disconnect_markers):
                                    disconnected_hit = True
                        except Exception:
                            pass

                    if self.json_path.exists():
                        try:
                            data = json.loads(self.json_path.read_text("utf-8"))
                            items = []
                            for it in data.get("impedances", []):
                                name = it.get("name", "")
                                val = float(it.get("value", 99999))
                                ok = bool(it.get("ok", (val != 99999 and val < 40.0)))
                                items.append({"name": name, "value": val, "ok": ok})
                            self.finished_items.emit(items)
                            return True, ""
                        except Exception as e:
                            last_log = f"JSON parse error: {e}"

                    try:
                        if self.status_path.exists():
                            s = self.status_path.read_text("utf-8").strip()
                            if s:
                                code = int(s.split(",", 1)[0])
                                msg = s.split(",", 1)[1] if "," in s else ""
                                # SDK ERROR
                                if code == 6:
                                    return False, (msg or "SDK error")
                                if code == 8:
                                    if "Impedance done" in msg and not self.json_path.exists():
                                        time.sleep(0.3)
                                        if self.json_path.exists():
                                            continue
                                        return False, "Impedance finished but no result file produced."
                                    return False, "disconnected::status_stopped"
                    except Exception:
                        pass

                    if rc is not None:
                        elapsed = time.time() - t0
                        prefix = "disconnected" if disconnected_hit else "early_exit"
                        return False, f"{prefix}::{elapsed:.2f}::{last_log or 'no-log'}"

                    if time.time() - t0 > self.timeout:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        return False, f"Timeout. Last log: {last_log or 'none'}"

                    time.sleep(0.08)

            def _clean_log(s: str) -> str:
                if not s:
                    return s
                noisy = ("bin_dir=", "cfg_dir=", "lic_dir=", "license_dir=")
                return "" if any(k in s for k in noisy) else s

            ok, msg = one_attempt()

            if not ok and isinstance(msg, str) and msg.startswith("disconnected::"):
                self.failed.emit(
                    "B-Alert ESU is not connected or access was denied. "
                    "Please check USB/power/Bluetooth, drivers, and permissions."
                )
                return

            if not ok and isinstance(msg, str) and msg.startswith("early_exit::"):
                parts = msg.split("::", 2)
                try:
                    elapsed = float(parts[1])
                except Exception:
                    elapsed = 0.0
                last_log = parts[2] if len(parts) > 2 else ""

                if elapsed < 2.5:
                    time.sleep(1.0)
                    ok2, msg2 = one_attempt()
                    if ok2:
                        return
                    if isinstance(msg2, str) and msg2.startswith("early_exit::"):
                        self.failed.emit("Please wait a few seconds and try again.")
                        return
                    msg = msg2 or msg
                    last_log = (msg.split("::", 2)[-1]
                                if isinstance(msg, str) and "::" in msg else last_log)

                clean = _clean_log(last_log)
                if clean:
                    self.failed.emit(f"Failed to measure impedance. Details: {clean}")
                else:
                    self.failed.emit("Failed to measure impedance. The SDK exited before producing results.")
                return

            if not ok:
                clean = _clean_log(str(msg))
                self.failed.emit(clean or "Failed to measure impedance.")
                return

            return

        except Exception as e:
            self.failed.emit(str(e))

# ----------------- table & canvas -----------------
class ImpTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Channel", "Value (kΩ)", ""])
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Expanding)

        hdr = self.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(2, 18)
        self.verticalHeader().setDefaultSectionSize(22)

    def _make_swatch(self, rgb):
        r, g, b = [int(x * 255) for x in rgb]
        box = QtWidgets.QLabel()
        box.setFixedSize(16, 16)
        box.setStyleSheet(
            f"background-color: rgb({r},{g},{b});"
            f"border: 1px solid #333; border-radius: 3px;"
        )
        wrapper = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(box)
        return wrapper

    def update_rows(self, items: List[dict]):
        self.setRowCount(0)
        rank = {name: i for i, name in enumerate(ORDER)}
        items = sorted(items, key=lambda x: rank.get(x.get("name", ""), 999))
        for it in items:
            n = it.get("name", "")
            v = float(it.get("value", 99999))
            ok = bool(it.get("ok", False))
            row = self.rowCount()
            self.insertRow(row)
            self.setItem(row, 0, QtWidgets.QTableWidgetItem(n))
            self.setItem(row, 1, QtWidgets.QTableWidgetItem("NA" if v == 99999 else f"{int(v)}"))
            self.setCellWidget(row, 2, self._make_swatch(color_for_value(v, ok)))
        self.resizeColumnsToContents()

class TopoCanvas(FigureCanvas):
    def __init__(self):
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        super().__init__(fig)
        self.ax = ax
        self._draw_base()

    def _draw_base(self):
        ax = self.ax
        ax.clear()
        gray = (0.5, 0.5, 0.5)
        R = HEAD_R

        head = plt.Circle((0.0, 0.0), R, fill=False, linewidth=2.0, color=gray)
        ax.add_patch(head)

        apex_y  = R + NOSE_APEX_H * R
        base_y  = R + 0.005 * R
        base_dx = NOSE_HALF * R
        nose = plt.Polygon(
            [(0.0, apex_y), (-base_dx, base_y), (base_dx, base_y)],
            closed=True, fill=False, linewidth=NOSE_LINE_W, color=gray
        )
        ax.add_patch(nose)

        ear_w = 2 * EAR_A * R
        ear_h = 2 * EAR_B * R
        left_center_x  = -R - EAR_A * R
        right_center_x =  R + EAR_A * R
        left_ear = Ellipse((left_center_x, 0.0), width=ear_w, height=ear_h,
                           angle=0, fill=False, linewidth=EAR_LINE_W, edgecolor=gray)
        right_ear = Ellipse((right_center_x, 0.0), width=ear_w, height=ear_h,
                            angle=0, fill=False, linewidth=EAR_LINE_W, edgecolor=gray)
        ax.add_patch(left_ear)
        ax.add_patch(right_ear)

        pad = PLOT_PAD * R
        ax.set_xlim(-R - pad, R + pad)
        ax.set_ylim(-R - pad, R + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    def update_impedance(self, items: list[dict]):
        self._draw_base()
        xs, ys, cols = [], [], []
        for it in items:
            n = it.get("name", "")
            if n not in POS:
                continue
            x, y = POS[n]
            v = float(it.get("value", 99999))
            ok = bool(it.get("ok", False))
            cols.append(color_for_value(v, ok))
            xs.append(x)
            ys.append(y)
            label = f"{n}\n{'NA' if v == 99999 else int(v)}"
            self.ax.text(x, y, label, ha="center", va="center", fontsize=8, zorder=3)
        self.ax.scatter(xs, ys, s=480, c=cols, edgecolors="k", linewidths=1, zorder=2)
        self.draw_idle()

# ----------------- options window -----------------
class BAlert_Options(BaseDeviceOptions):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setMinimumSize(WINDOW_MIN_W, WINDOW_MIN_H)
        self.resize(WINDOW_MIN_W, WINDOW_MIN_H)

        self.license_browse_btn.clicked.connect(self._on_browse_license)   # type: ignore[attr-defined]
        self.sync_license_btn.clicked.connect(self._on_sync_license)
        self.check_impedance_btn.clicked.connect(self._on_check_impedance) # type: ignore[attr-defined]

        self.license_sync_label = QtWidgets.QLabel(
            "License will be synced when you click 'Sync License' or start streaming."
        )
        self.license_sync_label.setWordWrap(True)
        self.license_sync_label.setStyleSheet("color:#9aa0a6;")

        self.status_label = QtWidgets.QLabel("")
        self.table = ImpTable(self)
        self.canvas = TopoCanvas()

        self.canvas.setMinimumSize(460, 460)
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.table)
        hbox.addWidget(self.canvas, 1)
        hbox.setStretch(0, 0)
        hbox.setStretch(1, 1)

        root_layout = self.layout() or QtWidgets.QVBoxLayout(self)
        if self.layout() is None:
            self.setLayout(root_layout)
        self._place_license_hint_above_check_button(root_layout)

        root_layout.addWidget(self.status_label)
        root_layout.addLayout(hbox)

        if getattr(AppConfigs(), "balert_license_dir", None):
            self.license_dir_lineedit.setText(AppConfigs().balert_license_dir) # type: ignore[attr-defined]

        empty = [{"name": ch, "value": 99999, "ok": False} for ch in ORDER]
        self.table.update_rows(empty)
        self.canvas.update_impedance(empty)

        self._worker: _ImpWorker | None = None

    def _place_license_hint_above_check_button(self, root_layout: QtWidgets.QVBoxLayout):
        row = None
        parent_w = self.check_impedance_btn.parentWidget()                  # type: ignore[attr-defined]
        if parent_w is not None and isinstance(parent_w.layout(), QtWidgets.QHBoxLayout):
            row = parent_w.layout()
        if row is not None:
            for i in range(root_layout.count()):
                it = root_layout.itemAt(i)
                if it and it.layout() is row:
                    root_layout.insertWidget(i, self.license_sync_label)
                    return
        root_layout.insertWidget(0, self.license_sync_label)

    # ----- license hint -----
    def _show_license_selected(self, _path: str):
        self.license_sync_label.setText("License folder selected.")
        self.license_sync_label.setStyleSheet("color:#9aa0a6;")

    def _show_license_synced(self):
        self.license_sync_label.setText("License synced successfully.")
        self.license_sync_label.setStyleSheet("color:#2e7d32; font-weight:600;")

    def _show_license_sync_failed(self, reason: str | None = None):
        msg = "License sync failed."
        if reason:
            msg += f" {reason}"
        self.license_sync_label.setText(msg)
        self.license_sync_label.setStyleSheet("color:#c62828; font-weight:600;")

    # ----- Sync License btn -----
    def _on_sync_license(self):
        exe = _balert_exe_path()
        if not exe.exists():
            QtWidgets.QMessageBox.critical(self, "Sync License", f"Cannot find BAlert.exe:\n{exe}")
            return

        lic_dir = (self.license_dir_lineedit.text() or "").strip()          # type: ignore[attr-defined]
        if not lic_dir or not Path(lic_dir).exists():
            QtWidgets.QMessageBox.critical(self, "Sync License", "Please select a valid License folder.")
            return

        _kill_existing_balert(timeout=1.0)
        self.sync_license_btn.setEnabled(False)
        try:
            _sync_license_to_exe_dir(exe, Path(lic_dir))
        except Exception as e:
            self._show_license_sync_failed("Please reselect the folder.")
            QtWidgets.QMessageBox.critical(self, "Sync License", f"License sync failed:\n{e}")
            return
        finally:
            QtCore.QThread.msleep(350)
            self.sync_license_btn.setEnabled(True)

        self._show_license_synced()
        try:
            AppConfigs().balert_license_dir = lic_dir
        except Exception:
            pass

    # ----- select license -----
    def _on_browse_license(self):
        saved_dir = getattr(AppConfigs(), "balert_license_dir", "")
        default_dir = saved_dir if saved_dir else str(Path.home())
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select License Folder",
            self.license_dir_lineedit.text() or default_dir                 # type: ignore[attr-defined]
        )
        if path:
            self.license_dir_lineedit.setText(path)                         # type: ignore[attr-defined]
            try:
                AppConfigs().balert_license_dir = path
            except Exception:
                pass
            self._show_license_selected(path)

    # ----- Check Impedance -----
    def _on_check_impedance(self):
        exe = _balert_exe_path()
        if not exe.exists():
            QtWidgets.QMessageBox.critical(self, "Check Impedance", f"Cannot find BAlert.exe:\n{exe}")
            return

        license_dir = (self.license_dir_lineedit.text() or "").strip()      # type: ignore[attr-defined]
        if not license_dir or not Path(license_dir).exists():
            QtWidgets.QMessageBox.critical(self, "Check Impedance", "Please select a valid License folder.")
            return

        _kill_existing_balert(timeout=1.0)
        try:
            lic_dst = _sync_license_to_exe_dir(exe, Path(license_dir))
        except Exception as e:
            self._show_license_sync_failed("Please reselect the folder.")
            QtWidgets.QMessageBox.critical(self, "Check Impedance", f"License sync failed:\n{e}")
            return
        QtCore.QThread.msleep(350)
        self._show_license_synced()

        try:
            AppConfigs().balert_license_dir = license_dir
        except Exception:
            pass

        self.check_btn_set_enabled(False)

        wd = _guess_work_dir_like_interface()
        self._worker = _ImpWorker(exe=exe, license_dir=str(lic_dst), work_dir=wd, timeout=60.0)
        self._worker.finished_items.connect(self._apply_impedance)
        self._worker.failed.connect(self._impedance_failed)
        self._worker.finished.connect(lambda: self.check_btn_set_enabled(True))
        self._worker.start()

    def check_btn_set_enabled(self, enabled: bool):
        try:
            self.check_impedance_btn.setEnabled(enabled)  # type: ignore[attr-defined]
            self.sync_license_btn.setEnabled(enabled)
            self.check_impedance_btn.setText("Check Impedance" if enabled else "Checking…")
        except Exception:
            pass

    # ----- impedance result -----
    def _apply_impedance(self, items: list):
        self.table.update_rows(items)
        self.canvas.update_impedance(items)
        eeg_items = [it for it in items if it.get("name") != "Ref"]
        need_fix = [
            it for it in eeg_items
            if float(it.get("value", 99999)) == 99999
               or float(it.get("value", 99999)) > 40.0
        ]
        total = 20
        self.status_label.setText("OK" if not need_fix else f"{len(need_fix)}/{total} channel(s) need fix")

    def _impedance_failed(self, msg: str):
        self.status_label.setText(f"Failed: {msg}")
        if "license" in msg.lower():
            self._show_license_sync_failed()
        QtWidgets.QMessageBox.critical(self, "Check Impedance", msg)

    # ----------
    def start_stream_args(self) -> dict:
        license_dir = (self.license_dir_lineedit.text() or "").strip()      # type: ignore[attr-defined]
        exe = _balert_exe_path()
        return {"license_dir": license_dir, "exe_path": str(exe)}