# -*- coding: utf-8 -*-
"""
CameraRecorderGUI (responsive preview, segment-synced audio, min-duration filter)
- Preview canvas resizes to chosen recording resolution (and scales with window)
- Fresh WAV per segment; start/stop exactly with MP4 -> reliable sound
- Skip segments shorter than 3.0s (delete mp4/wav, no mux)
- Motion boxes (toggle), flicker-free preview, background mux (no cmd flash)
- Camera names on Windows via ffmpeg dshow listing (if ffmpeg.exe present)
- Mic level meter + Test Mic
- Log panel (log.txt) with checkbox
"""

import os
import sys
import cv2
import time
import wave
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image, ImageTk

# ---- optional: sounddevice
try:
    import sounddevice as sd
    HAVE_SD = True
except Exception:
    HAVE_SD = False

APP_NAME = "CameraRecorderGUI"
LOG_FILE = "log.txt"
RECORD_DIR = "recordings"
MIN_SEG_SECONDS = 3.0  # <---- do not keep clips shorter than this
os.makedirs(RECORD_DIR, exist_ok=True)

# ---------------- Logging ----------------


class UiLogger:
    def __init__(self, tk_root_getter):
        self._file_lock = threading.Lock()
        self._tk_root_getter = tk_root_getter

    def write_line(self, level: str, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        with self._file_lock:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        root = self._tk_root_getter()
        if root is not None:
            try:
                root.event_generate("<<APP_LOG>>", when="tail")
            except Exception:
                pass
        print(line)


LOGGER = UiLogger(lambda: None)


def log(level: str, msg: str):
    LOGGER.write_line(level, msg)


def ts_name(prefix: str, ext: str) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(RECORD_DIR, f"{prefix}_{stamp}.{ext}")


def has_ffmpeg() -> Optional[str]:
    candidates = []
    exe_dir = os.path.dirname(sys.executable) if getattr(
        sys, 'frozen', False) else os.getcwd()
    candidates.append(os.path.join(exe_dir, "ffmpeg.exe"))
    candidates.append("ffmpeg.exe")
    candidates.append("ffmpeg")
    for c in candidates:
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            p = subprocess.run([c, "-version"], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, creationflags=flags)
            if p.returncode == 0:
                return c
        except Exception:
            pass
    return None


def list_windows_dshow_cameras(ffmpeg_path: Optional[str]) -> List[str]:
    if os.name != "nt" or not ffmpeg_path:
        return []
    try:
        flags = subprocess.CREATE_NO_WINDOW
        p = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-f", "dshow",
                "-list_devices", "true", "-i", "dummy"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags,
            text=True, encoding="utf-8", errors="ignore"
        )
        names = []
        for line in p.stderr.splitlines():
            line = line.strip()
            if line.startswith('"') and line.endswith('"') and len(line) > 2:
                names.append(line.strip('"'))
        out, seen = [], set()
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out
    except Exception as e:
        log("WARN", f"ffmpeg dshow list failed: {e}")
        return []

# ---------------- Audio ----------------


@dataclass
class AudioConfig:
    device: Optional[int] = None
    channels: int = 1
    latency: str = "low"
    samplerate: int = 44100
    dtype: str = "int16"


class AudioRecorder:
    def __init__(self, cfg: AudioConfig, meter_cb):
        self.cfg = cfg
        self._meter = meter_cb
        self.stream: Optional[sd.InputStream] = None if HAVE_SD else None
        self._wav: Optional[wave.Wave_write] = None
        self._q: queue.Queue = queue.Queue(maxsize=100)
        self._drainer: Optional[threading.Thread] = None
        self._active = threading.Event()

    def _callback(self, indata, frames, time_info, status):
        if status:
            log("WARN", f"audio status: {status}")
        try:
            if self._meter:
                if indata.dtype.kind == 'i':
                    peak = np.abs(indata.astype(np.int32)).max() / 32767.0
                else:
                    peak = float(np.abs(indata).max())
                self._meter(min(1.0, float(peak)))
        except Exception:
            pass
        try:
            self._q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def open(self) -> Tuple[bool, str]:
        if not HAVE_SD:
            return False, "sounddevice not installed"
        try:
            sr = 44100
            if self.cfg.device is not None:
                di = sd.query_devices(self.cfg.device)
                sr = int(di.get("default_samplerate") or sr)
            self.cfg.samplerate = sr
            kwargs = dict(
                samplerate=sr, channels=self.cfg.channels, dtype=self.cfg.dtype,
                latency=self.cfg.latency, callback=self._callback
            )
            if self.cfg.device is not None:
                kwargs["device"] = self.cfg.device
            st = sd.InputStream(**kwargs)
            st.start()
            self.stream = st
            log("AUDIO",
                f"opened: dev={self.cfg.device}, {sr}Hz, ch={self.cfg.channels}, lat={self.cfg.latency}, dtype={self.cfg.dtype}")
            return True, ""
        except Exception as e:
            return False, str(e)

    def start_segment(self, wav_path: str):
        self.stop_segment()
        wf = wave.open(wav_path, "wb")
        wf.setnchannels(self.cfg.channels)
        wf.setsampwidth(2 if self.cfg.dtype == "int16" else 4)
        wf.setframerate(self.cfg.samplerate)
        self._wav = wf
        self._active.set()
        self._drainer = threading.Thread(target=self._drain, daemon=True)
        self._drainer.start()

    def _drain(self):
        try:
            while self._active.is_set():
                try:
                    chunk = self._q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if self._wav:
                    self._wav.writeframes(chunk)
        except Exception as e:
            log("ERROR", f"audio drain: {e}")

    def stop_segment(self):
        self._active.clear()
        if self._drainer and self._drainer.is_alive():
            self._drainer.join(timeout=1.0)
        self._drainer = None
        if self._wav:
            try:
                self._wav.close()
            except Exception:
                pass
        self._wav = None
        # flush any leftovers
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except Exception:
                break

    def close(self):
        self.stop_segment()
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.stream = None

# ---------------- Camera Worker ----------------


@dataclass
class CamConfig:
    backend: str = "MSMF"     # AUTO|MSMF|DSHOW
    index: int = 0
    width: int = 1280
    height: int = 720
    mode_motion: bool = True
    show_motion_boxes: bool = True
    motion_thresh: int = 25
    motion_min_area: int = 2500
    preview_fps_limit: int = 15


class CameraWorker(threading.Thread):
    def __init__(
        self,
        cfg: CamConfig,
        on_preview,                 # frame BGR -> UI
        # base path (without extension) -> app starts WAV
        on_segment_begin,
        on_segment_end,             # mp4_path, duration_sec -> app stops WAV & mux/cleanup
        stop_evt: threading.Event
    ):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.on_preview = on_preview
        self.on_segment_begin = on_segment_begin
        self.on_segment_end = on_segment_end
        self.stop_evt = stop_evt

        self._vw: Optional[cv2.VideoWriter] = None
        self._bg = None
        self.fps = 15.0
        self._segment_start_ts = 0.0
        self._recording = False

    def _open_cap(self) -> Optional[cv2.VideoCapture]:
        api = 0
        if self.cfg.backend == "MSMF":
            api = cv2.CAP_MSMF
        elif self.cfg.backend == "DSHOW":
            api = cv2.CAP_DSHOW
        cap = cv2.VideoCapture(self.cfg.index if api ==
                               0 else (self.cfg.index + api))
        # request size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            # estimate quickly
            t0, n, cnt = time.time(), 8, 0
            while cnt < n:
                ok, _ = cap.read()
                if not ok:
                    break
                cnt += 1
            dt = max(1e-6, time.time() - t0)
            fps = cnt / dt
        self.fps = float(max(5.0, min(60.0, fps)))
        log("INFO",
            f"Camera opened via {self.cfg.backend}: {int(cap.get(3))}x{int(cap.get(4))} @ ~{self.fps:.2f}fps (using {self.fps:.2f})")
        return cap

    def _start_segment(self):
        base = os.path.splitext(
            ts_name("motion" if self.cfg.mode_motion else "normal", "mp4"))[0]
        mp4_path = base + ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._vw = cv2.VideoWriter(
            mp4_path, fourcc, self.fps, (self.cfg.width, self.cfg.height))
        self._segment_start_ts = time.time()
        self._recording = True
        # inform app BEFORE first frame is written, so audio aligns
        self.on_segment_begin(base)
        log("REC", f"Started: {os.path.basename(mp4_path)}")

    def _stop_segment(self):
        if not self._recording:
            return
        try:
            if self._vw:
                self._vw.release()
        except Exception:
            pass
        self._vw = None
        dur = max(0.0, time.time() - self._segment_start_ts)
        # base path we used for this seg:
        # derive from REC log? Better: keep last base path; reconstruct from timestamp
        # Easiest: return last written path via VideoWriter name – not available; so rebuild:
        # We'll rely on on_segment_begin storing "current_base" for mux. So here just signal end.
        self.on_segment_end(dur)
        self._recording = False

    def run(self):
        cap = self._open_cap()
        if cap is None:
            log("ERROR", "Failed to open camera.")
            return
        self._bg = None
        next_preview = 0.0
        motion_hold_until = 0.0
        hold_secs = 1.5

        while not self.stop_evt.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.resize(
                frame, (self.cfg.width, self.cfg.height), interpolation=cv2.INTER_AREA)
            preview = frame.copy()

            has_motion = True
            boxes = []
            if self.cfg.mode_motion:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 3)
                if self._bg is None:
                    self._bg = gray.astype("float")
                cv2.accumulateWeighted(gray, self._bg, 0.01)
                delta = cv2.absdiff(gray, cv2.convertScaleAbs(self._bg))
                thresh = cv2.threshold(
                    delta, self.cfg.motion_thresh, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                cnts, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                has_motion = False
                for c in cnts:
                    if cv2.contourArea(c) < self.cfg.motion_min_area:
                        continue
                    (x, y, w, h) = cv2.boundingRect(c)
                    boxes.append((x, y, w, h))
                has_motion = len(boxes) > 0

                if self.cfg.show_motion_boxes and boxes:
                    for (x, y, w, h) in boxes:
                        cv2.rectangle(preview, (x, y),
                                      (x + w, y + h), (0, 255, 0), 2)

            now = time.time()
            if self.cfg.mode_motion:
                if has_motion:
                    motion_hold_until = now + hold_secs
                if has_motion or now < motion_hold_until:
                    if not self._recording:
                        self._start_segment()
                else:
                    if self._recording:
                        self._stop_segment()
            else:
                if not self._recording:
                    self._start_segment()

            if self._recording and self._vw:
                self._vw.write(frame)

            if now >= next_preview:
                self.on_preview(preview)
                next_preview = now + 1.0 / max(1, self.cfg.preview_fps_limit)

        if self._recording:
            self._stop_segment()
        cap.release()
        log("INFO", "Worker stopped.")

# ---------------- App (Tk) ----------------


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        global LOGGER
        LOGGER = UiLogger(lambda: self)

        self.title(APP_NAME)
        self.geometry("1100x760")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.ffmpeg_path = has_ffmpeg()
        if self.ffmpeg_path:
            log("INFO", f"ffmpeg found: {self.ffmpeg_path}")
        else:
            log("WARN", "ffmpeg not found; camera names (Win) & mux disabled.")

        # vars
        self.var_backend = tk.StringVar(
            value="MSMF" if os.name == "nt" else "AUTO")
        self.var_cam = tk.StringVar(value="")
        self.var_res = tk.StringVar(value="1280x720")
        self.var_mode = tk.StringVar(value="MOTION")
        self.var_motion_boxes = tk.BooleanVar(value=True)
        self.var_motion_thresh = tk.IntVar(value=25)
        self.var_motion_area = tk.IntVar(value=2500)
        self.var_audio_enable = tk.BooleanVar(value=True)
        self.var_audio_latency = tk.StringVar(value="low")
        self.var_audio_dev = tk.StringVar(value="")
        self.var_show_log = tk.BooleanVar(value=False)

        self.log_toggle_var = self.var_show_log

        # app state
        self._stop_evt = threading.Event()
        self._worker: Optional[CameraWorker] = None
        self._preview_q: queue.Queue = queue.Queue(maxsize=2)
        self._preview_imgtk = None
        self._mux_q: queue.Queue = queue.Queue()
        self._current_seg_base: Optional[str] = None  # set on seg begin
        self._current_seg_mp4: Optional[str] = None   # derived from base
        self._current_seg_wav: Optional[str] = None   # derived from base
        self.audio_rec: Optional[AudioRecorder] = None

        self._build_ui()
        self._list_cameras()
        self._list_audio()

        self.after(15, self._drain_preview)
        threading.Thread(target=self._mux_worker, daemon=True).start()
        self.bind("<<APP_LOG>>", self._ui_log_sink, add="+")

        log("START", "Application started")

    # ---------- UI ----------
    def _build_ui(self):
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, padding=8)
        right = ttk.Frame(paned, padding=8)
        paned.add(left, weight=1)
        paned.add(right, weight=2)

        row = 0
        ttk.Label(left, text="Camera backend:").grid(
            row=row, column=0, sticky="w")
        ttk.Combobox(left, textvariable=self.var_backend, values=[
                     "AUTO", "MSMF", "DSHOW"], width=10, state="readonly").grid(row=row, column=1, sticky="w", padx=4)
        row += 1

        ttk.Label(left, text="Camera:").grid(row=row, column=0, sticky="w")
        self.cb_cam = ttk.Combobox(
            left, textvariable=self.var_cam, width=45, state="readonly")
        self.cb_cam.grid(row=row, column=1, sticky="we", padx=4)
        row += 1

        ttk.Label(left, text="Resolution:").grid(row=row, column=0, sticky="w")
        ttk.Combobox(left, textvariable=self.var_res, values=["1920x1080", "1600x900", "1280x720", "1024x576", "960x540",
                     "848x480", "640x480", "640x360"], state="readonly", width=12).grid(row=row, column=1, sticky="w", padx=4)
        row += 1

        ttk.Label(left, text="Mode:").grid(row=row, column=0, sticky="w")
        frm = ttk.Frame(left)
        frm.grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Radiobutton(frm, text="Motion detect",
                        variable=self.var_mode, value="MOTION").pack(side=tk.LEFT)
        ttk.Radiobutton(frm, text="Normal", variable=self.var_mode,
                        value="NORMAL").pack(side=tk.LEFT)

        ttk.Checkbutton(left, text="Show motion boxes", variable=self.var_motion_boxes).grid(
            row=row, column=1, sticky="w")
        row += 1
        ttk.Label(left, text="Motion threshold:").grid(
            row=row, column=0, sticky="w")
        ttk.Scale(left, from_=5, to=60, variable=self.var_motion_thresh,
                  orient=tk.HORIZONTAL).grid(row=row, column=1, sticky="we")
        row += 1
        ttk.Label(left, text="Min area (px):").grid(
            row=row, column=0, sticky="w")
        ttk.Scale(left, from_=500, to=12000, variable=self.var_motion_area,
                  orient=tk.HORIZONTAL).grid(row=row, column=1, sticky="we")
        row += 1

        ttk.Checkbutton(left, text="Record audio", variable=self.var_audio_enable).grid(
            row=row, column=1, sticky="w")
        row += 1
        ttk.Label(left, text="Audio device:").grid(
            row=row, column=0, sticky="w")
        self.cb_audio = ttk.Combobox(
            left, textvariable=self.var_audio_dev, width=45, state="readonly")
        self.cb_audio.grid(row=row, column=1, sticky="we", padx=4)
        row += 1

        ttk.Label(left, text="Audio latency:").grid(
            row=row, column=0, sticky="w")
        ttk.Combobox(left, textvariable=self.var_audio_latency, values=[
                     "low", "high"], state="readonly", width=10).grid(row=row, column=1, sticky="w", padx=4)
        row += 1

        ttk.Label(left, text="Mic level:").grid(row=row, column=0, sticky="w")
        self.pb_level = ttk.Progressbar(left, maximum=100, length=200)
        self.pb_level.grid(row=row, column=1, sticky="w", pady=2)
        row += 1
        ttk.Button(left, text="Test Mic (3s)", command=self.on_test_mic).grid(
            row=row, column=1, sticky="w")
        row += 1

        ttk.Checkbutton(left, text="Show log panel (log.txt)", variable=self.var_show_log,
                        command=self._toggle_log).grid(row=row, column=1, sticky="w")
        row += 1

        frb = ttk.Frame(left)
        frb.grid(row=row, column=0, columnspan=2, pady=10, sticky="we")
        row += 1
        self.bt_start = ttk.Button(frb, text="Start", command=self.on_start)
        self.bt_stop = ttk.Button(
            frb, text="Stop", command=self.on_stop, state="disabled")
        self.bt_start.pack(side=tk.LEFT, padx=4)
        self.bt_stop.pack(side=tk.LEFT, padx=4)
        self.lbl_status = ttk.Label(frb, text="Idle")
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        # Preview canvas — we’ll size it to recording resolution on Start
        self.cv = tk.Canvas(right, bg="black",
                            highlightthickness=1, highlightbackground="#444")
        self.cv.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(right, height=12, state=tk.DISABLED)
        # not packed until toggled

        for c in range(2):
            left.grid_columnconfigure(c, weight=1)
        right.pack_propagate(False)

    # ---------- Devices ----------
    def _list_cameras(self):
        items = []
        names = list_windows_dshow_cameras(
            self.ffmpeg_path) if os.name == "nt" else []
        found = 0
        backend = self.var_backend.get()
        for idx in range(0, 8):
            api = 0
            if backend == "MSMF":
                api = cv2.CAP_MSMF
            elif backend == "DSHOW":
                api = cv2.CAP_DSHOW
            cap = cv2.VideoCapture(idx if api == 0 else (idx + api))
            ok, _ = cap.read()
            cap.release()
            if ok:
                nm = names[idx] if idx < len(names) else f"Camera {idx}"
                items.append(f"{idx} - {nm}")
                found += 1
        if not items:
            items = ["0 - Camera 0"]
        self.cb_cam["values"] = items
        self.var_cam.set(items[0])
        log("INFO", f"Found {found} camera(s).")

    def _list_audio(self):
        if not HAVE_SD:
            self.cb_audio["values"] = ["(sounddevice not installed)"]
            self.var_audio_dev.set("(sounddevice not installed)")
            return
        devs = sd.query_devices()
        inputs = []
        for i, d in enumerate(devs):
            if d["max_input_channels"] > 0:
                host = sd.query_hostapis(d["hostapi"])["name"]
                inputs.append(f"{i} - {d['name']} ({host})")
        if inputs:
            self.cb_audio["values"] = inputs
            self.var_audio_dev.set(inputs[0])
        else:
            self.cb_audio["values"] = ["(no input devices)"]
            self.var_audio_dev.set("(no input devices)")
        log("INFO", "Audio devices refreshed.")

    # ---------- Mic test ----------
    def _level_cb(self, peak01: float):
        self.pb_level["value"] = int(peak01 * 100)

    def on_test_mic(self):
        if not HAVE_SD:
            messagebox.showwarning("Audio", "sounddevice not installed.")
            return
        dev_idx = None
        s = self.var_audio_dev.get()
        if s and " - " in s and s[0].isdigit():
            try:
                dev_idx = int(s.split(" - ", 1)[0])
            except Exception:
                dev_idx = None

        cfg = AudioConfig(device=dev_idx, channels=1,
                          latency=self.var_audio_latency.get(), dtype="int16")
        ar = AudioRecorder(cfg, self._level_cb)
        ok, err = ar.open()
        log("TEST", "Opening mic for test...")
        if not ok:
            log("TEST", f"Mic test failed: {err}")
            messagebox.showerror("Audio Test", f"Could not open mic:\n{err}")
            return
        tmp_wav = ts_name("mic_test", "wav")
        ar.start_segment(tmp_wav)
        self.after(3000, lambda: self._finish_test(ar, tmp_wav))

    def _finish_test(self, ar: AudioRecorder, wav_path: str):
        ar.stop_segment()
        ar.close()
        log("TEST", "Mic test complete.")
        messagebox.showinfo(
            "Audio Test", f"Mic OK.\nSaved short sample:\n{wav_path}")

    # ---------- Start / Stop ----------
    def _parse_res(self) -> Tuple[int, int]:
        try:
            w, h = self.var_res.get().split("x")
            return int(w), int(h)
        except Exception:
            return 1280, 720

    def on_start(self):
        if self._worker:
            return
        # camera
        cam_sel = self.var_cam.get()
        try:
            cam_idx = int(cam_sel.split(" - ", 1)[0])
        except Exception:
            cam_idx = 0
        w, h = self._parse_res()

        # resize preview canvas EXACTLY to recording size
        self.cv.config(width=w, height=h)

        cfg = CamConfig(
            backend=self.var_backend.get(),
            index=cam_idx, width=w, height=h,
            mode_motion=(self.var_mode.get() == "MOTION"),
            show_motion_boxes=self.var_motion_boxes.get(),
            motion_thresh=self.var_motion_thresh.get(),
            motion_min_area=self.var_motion_area.get(),
            preview_fps_limit=15
        )

        # audio
        self.audio_rec = None
        if self.var_audio_enable.get() and HAVE_SD:
            dev_idx = None
            s = self.var_audio_dev.get()
            if s and " - " in s and s[0].isdigit():
                try:
                    dev_idx = int(s.split(" - ", 1)[0])
                except Exception:
                    dev_idx = None
            acfg = AudioConfig(device=dev_idx, channels=1,
                               latency=self.var_audio_latency.get(), dtype="int16")
            ar = AudioRecorder(acfg, self._level_cb)
            ok, err = ar.open()
            if ok:
                self.audio_rec = ar
            else:
                log("WARN", f"Audio disabled: {err}")
                messagebox.showwarning("Audio", f"Audio disabled:\n{err}")

        self._stop_evt.clear()
        self._current_seg_base = None
        self._current_seg_mp4 = None
        self._current_seg_wav = None

        self._worker = CameraWorker(
            cfg,
            on_preview=self._enqueue_preview,
            on_segment_begin=self._on_segment_begin,
            on_segment_end=self._on_segment_end,
            stop_evt=self._stop_evt
        )
        self._worker.start()

        self.bt_start.config(state="disabled")
        self.bt_stop.config(state="normal")
        self.lbl_status.config(
            text=f"Recording ({'motion' if cfg.mode_motion else 'normal'})")
        log("INFO", f"Recorder started. Preview target={w}x{h}")

    def on_stop(self):
        if self._worker:
            self._stop_evt.set()
            self._worker.join(timeout=5.0)
            self._worker = None
        if self.audio_rec:
            # if a segment is active, worker already told us to stop it
            self.audio_rec.close()
            self.audio_rec = None
        self.bt_start.config(state="normal")
        self.bt_stop.config(state="disabled")
        self.lbl_status.config(text="Stopped")
        log("INFO", "Stopped.")

    # ---------- Preview ----------
    def _enqueue_preview(self, frame_bgr: np.ndarray):
        try:
            if not self._preview_q.empty():
                self._preview_q.get_nowait()
            self._preview_q.put_nowait(frame_bgr)
        except queue.Full:
            pass

    def _drain_preview(self):
        frame = None
        try:
            frame = self._preview_q.get_nowait()
        except queue.Empty:
            pass
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            # scale to current canvas size (which we set to recording size, but allow responsiveness)
            cw = max(1, self.cv.winfo_width())
            ch = max(1, self.cv.winfo_height())
            im = im.resize((cw, ch), Image.BILINEAR)
            self._preview_imgtk = ImageTk.PhotoImage(im)
            self.cv.delete("all")
            self.cv.create_image(0, 0, image=self._preview_imgtk, anchor="nw")
        self.after(15, self._drain_preview)

    # ---------- Segment lifecycle from worker ----------
    def _on_segment_begin(self, base_path_no_ext: str):
        self._current_seg_base = base_path_no_ext
        self._current_seg_mp4 = base_path_no_ext + ".mp4"
        self._current_seg_wav = base_path_no_ext + ".wav"
        if self.audio_rec:
            self.audio_rec.start_segment(self._current_seg_wav)

    def _on_segment_end(self, duration_sec: float):
        # stop audio for this segment
        if self.audio_rec:
            self.audio_rec.stop_segment()
        mp4 = self._current_seg_mp4
        wav = self._current_seg_wav
        self._current_seg_base = None
        self._current_seg_mp4 = None
        self._current_seg_wav = None

        # enforce min duration
        if duration_sec < MIN_SEG_SECONDS:
            log("INFO", f"Segment < {MIN_SEG_SECONDS:.1f}s -> discard")
            # delete files if exist
            try:
                if wav and os.path.exists(wav):
                    os.remove(wav)
            except Exception:
                pass
            try:
                if mp4 and os.path.exists(mp4):
                    os.remove(mp4)
            except Exception:
                pass
            return

        # queue mux if we have both and ffmpeg
        if self.ffmpeg_path and wav and os.path.exists(wav) and mp4 and os.path.exists(mp4):
            self._mux_q.put((mp4, wav))
        else:
            log("INFO", "Mux skipped (missing ffmpeg or audio).")

    # ---------- Mux worker ----------
    def _mux_worker(self):
        while True:
            try:
                mp4, wav = self._mux_q.get()
            except Exception:
                time.sleep(0.1)
                continue
            out = os.path.splitext(mp4)[0] + ".av.mp4"
            cmd = [
                self.ffmpeg_path, "-y",
                "-i", mp4,
                "-i", wav,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "160k",
                "-shortest", out
            ]
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            log("MUX", " ".join(cmd))
            try:
                p = subprocess.run(cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, creationflags=flags)
                if p.returncode != 0:
                    err = p.stderr.decode("utf-8", errors="ignore")[:4000]
                    log("ERROR", f"mux failed: {err}")
                else:
                    log("INFO", f"Muxed: {os.path.basename(out)}")
                    # optional: delete raw mp4/wav after success
                    try:
                        os.remove(mp4)
                    except Exception:
                        pass
                    try:
                        os.remove(wav)
                    except Exception:
                        pass
            except Exception as e:
                log("ERROR", f"mux exception: {e}")

    # ---------- Log UI ----------
    def _toggle_log(self):
        if self.var_show_log.get():
            self.log_text.pack(fill=tk.X, side=tk.BOTTOM, pady=4)
            self._append_log_tail()
        else:
            try:
                self.log_text.pack_forget()
            except Exception:
                pass

    def _append_log_tail(self, lines=200):
        if not self.var_show_log.get():
            return
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                tail = f.readlines()[-lines:]
        except Exception:
            tail = []
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, "".join(tail))
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def _ui_log_sink(self, _evt):
        if getattr(self, "log_toggle_var", None) and self.log_toggle_var.get():
            self._append_log_tail(200)

    # ---------- Close ----------
    def on_close(self):
        log("STOP", "Application closing")
        try:
            self.on_stop()
        except Exception:
            pass
        self.destroy()


# -------------- entry --------------
if __name__ == "__main__":
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [START] {APP_NAME} started\n")
    except Exception:
        pass
    app = App()
    app.mainloop()
