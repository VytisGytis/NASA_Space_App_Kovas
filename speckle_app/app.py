"""
Speckle Lab - CustomTkinter GUI for capturing and comparing laser speckle scans.

Run from the project root:   python -m speckle_app
"""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import OrderedDict
from pathlib import Path
import csv
from tkinter import filedialog

import customtkinter as ctk
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib import colormaps  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.widgets import RectangleSelector  # noqa: E402

from . import processing as P  # noqa: E402
from .capture import DEFAULTS as CAPTURE_DEFAULTS  # noqa: E402
from .capture import Capture, detect_esp_ip  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"

# ---------------------------------------------------------------- palette
BG = "#0e1014"
PANEL = "#151820"
CARD = "#1b1f29"
CARD_HI = "#232837"
BORDER = "#2a3040"
TEXT = "#e7e9ef"
MUTED = "#8b93a7"
ACCENT = "#5b8cff"
ACCENT_HOVER = "#4a77e0"
LASER = "#ff4d6d"
LASER_HOVER = "#e03a5a"
CMP_COLOR = "#ff9f43"
OK = "#3ddc97"
WARN = "#ffb547"
BAD = "#ff5c7a"

VIEWS = ["Overlay", "Reference", "Compare", "K ref", "K cmp", "ΔK/K",
         "Correlation", "Flow ratio", "Temporal K"]
SINGLE_VIEWS = ["Overlay", "Image", "K map", "Surroundings K", "ΔK/K", "Flow"]
VIEW_ALIAS = {"Image": "Reference", "K map": "K ref", "Surroundings K": "K cmp",
              "Flow": "Flow ratio"}
MODES = ["Single image", "Compare scans"]
METHODS = {"Combined": "combined", "Contrast": "contrast", "Correlation": "correlation"}
MAX_DISPLAY_PX = 1400


def load_settings() -> dict:
    s = {"incoming": str(ROOT / "incoming"), "capture": dict(CAPTURE_DEFAULTS),
         "analysis": {"channel": "Auto", "rotate": 0, "win": 7, "align": True,
                      "method": "Combined", "k_thr": 20, "corr_thr": 0.5,
                      "min_area": 1.0, "auto": True, "mode": "Single image"}}
    try:
        saved = json.loads(SETTINGS_FILE.read_text())
        s["incoming"] = saved.get("incoming", s["incoming"])
        s["capture"].update(saved.get("capture", {}))
        s["analysis"].update(saved.get("analysis", {}))
    except (OSError, ValueError):
        pass
    return s


def font(size=13, weight="normal"):
    return ctk.CTkFont(size=size, weight=weight)


# ---------------------------------------------------------------- widgets
class StatCard(ctk.CTkFrame):
    def __init__(self, master, title):
        super().__init__(master, fg_color=CARD, corner_radius=12,
                         border_width=1, border_color=BORDER)
        self.title = ctk.CTkLabel(self, text=title.upper(), font=font(10, "bold"),
                                  text_color=MUTED)
        self.title.pack(anchor="w", padx=14, pady=(10, 0))
        self.value = ctk.CTkLabel(self, text="—", font=font(22, "bold"), text_color=TEXT)
        self.value.pack(anchor="w", padx=14)
        self.sub = ctk.CTkLabel(self, text=" ", font=font(11), text_color=MUTED)
        self.sub.pack(anchor="w", padx=14, pady=(0, 10))

    def set(self, value, sub=" ", color=TEXT, title=None):
        if title:
            self.title.configure(text=title.upper())
        self.value.configure(text=value, text_color=color)
        self.sub.configure(text=sub)


class ScanRow(ctk.CTkFrame):
    def __init__(self, master, app, path: Path, n: int, complete: bool, external: bool):
        is_ref, is_cmp = path == app.ref, path == app.cmp and not app.single
        border = ACCENT if is_ref else CMP_COLOR if is_cmp else BORDER
        super().__init__(master, fg_color=CARD_HI if (is_ref or is_cmp) else CARD,
                         corner_radius=10, border_width=1, border_color=border)
        self.columnconfigure(0, weight=1)
        name = ("↗ " if external else "") + path.name
        ctk.CTkLabel(self, text=name, font=font(13, "bold"), text_color=TEXT,
                     anchor="w").grid(row=0, column=0, sticky="w", padx=(12, 4), pady=(8, 0))
        when = time.strftime("%H:%M", time.localtime(path.stat().st_mtime))
        badge, color = ("✓ done", OK) if complete else ("⚠ partial", WARN)
        if external:
            badge, color = ("image", MUTED) if path.is_file() else \
                ("images", MUTED) if isinstance(path, P.FileSet) else ("folder", MUTED)
        info = ctk.CTkFrame(self, fg_color="transparent")
        info.grid(row=1, column=0, sticky="w", padx=(12, 4), pady=(0, 8))
        single = app.single
        count = "1 image" if path.is_file() else f"{n} image{'s' * (n != 1)}"
        ctk.CTkLabel(info, text=f"{count} · {when}  ",
                     font=font(11), text_color=MUTED).pack(side="left")
        ctk.CTkLabel(info, text=badge, font=font(11), text_color=color).pack(side="left")

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.grid(row=0, column=1, rowspan=2, padx=(0, 8))
        roles = [("USE" if single else "REF", is_ref, ACCENT, ACCENT_HOVER, "ref")]
        if not single:
            roles.append(("CMP", is_cmp, CMP_COLOR, "#e08a35", "cmp"))
        for text, on, col, hov, role in roles:
            ctk.CTkButton(btns, text=text, width=46, height=26, corner_radius=8,
                          font=font(11, "bold"),
                          fg_color=col if on else "transparent", hover_color=hov,
                          border_width=1, border_color=col,
                          text_color=BG if on else col,
                          command=lambda r=role: app.pick(path, r)).pack(side="left", padx=2)


# ---------------------------------------------------------------- app
class App(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        super().__init__(fg_color=BG)
        self.title("Speckle Lab")
        self.geometry("1520x940")
        self.minsize(1180, 740)

        self.settings = load_settings()
        self.q: queue.Queue = queue.Queue()
        self.cache: OrderedDict = OrderedDict()
        self.ref: Path | None = None
        self.cmp: Path | None = None
        self.extra_dirs: list[Path] = []
        self.comp: P.Comparison | None = None
        self.verdict: P.Verdict | None = None
        self.view = "Overlay"
        self.roi = None                      # (x0, x1, y0, y1) in full-res pixels
        self.busy = False
        self.ax = None
        self.selector = None
        self._pan = None
        self._eval_job = None
        self._lims = None
        self.frame_idx = 0
        self.capture = Capture(self.settings["capture"], self.settings["incoming"],
                               status=lambda s: self._ui(self.set_status, s))

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

        self.known = set(self._scan_dirs())
        self.waiting: set[Path] = set()
        self.set_mode(self.mode_var.get())

        self.bind("<Return>", lambda e: self.analyze())
        self.bind("<F5>", lambda e: self.refresh_scans())
        self.bind("<Left>", lambda e: self._arrow(e, -1))
        self.bind("<Right>", lambda e: self._arrow(e, 1))
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(50, self._pump)
        self.after(2000, self._watch)

    # ============================================================ layout
    def _section(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=font(11, "bold"),
                     text_color=MUTED).pack(anchor="w", padx=18, pady=(14, 4))

    def _build_sidebar(self):
        a = self.settings["analysis"]
        sb = ctk.CTkScrollableFrame(self, width=330, fg_color=PANEL, corner_radius=0,
                                    scrollbar_button_color=BORDER,
                                    scrollbar_button_hover_color=MUTED)
        sb.grid(row=0, column=0, sticky="nsew")

        head = ctk.CTkFrame(sb, fg_color="transparent")
        head.pack(fill="x", padx=18, pady=(18, 0))
        ctk.CTkLabel(head, text="●", font=font(26), text_color=LASER).pack(side="left")
        ctk.CTkLabel(head, text=" Speckle Lab", font=font(24, "bold"),
                     text_color=TEXT).pack(side="left")
        ctk.CTkLabel(sb, text="laser speckle change detection", font=font(12),
                     text_color=MUTED).pack(anchor="w", padx=20)

        # ---- capture
        self._section(sb, "CAPTURE")
        row = ctk.CTkFrame(sb, fg_color="transparent")
        row.pack(fill="x", padx=18)
        self.ip_var = ctk.StringVar(value=self.settings["capture"].get("esp_ip", ""))
        ctk.CTkEntry(row, textvariable=self.ip_var, placeholder_text="ESP32 IP",
                     height=34, fg_color=CARD, border_color=BORDER).pack(
            side="left", fill="x", expand=True)
        self.detect_btn = ctk.CTkButton(row, text="Detect", width=74, height=34,
                                        fg_color=CARD_HI, hover_color=BORDER,
                                        command=self.detect_ip)
        self.detect_btn.pack(side="left", padx=(8, 0))

        self.scan_btn = ctk.CTkButton(sb, text="SCAN", height=52, corner_radius=12,
                                      font=font(20, "bold"), fg_color=LASER,
                                      hover_color=LASER_HOVER, command=self.scan)
        self.scan_btn.pack(fill="x", padx=18, pady=(10, 6))
        row = ctk.CTkFrame(sb, fg_color="transparent")
        row.pack(fill="x", padx=18)
        self.retry_btn = ctk.CTkButton(row, text="Retry pulls", height=30, state="disabled",
                                       fg_color=CARD_HI, hover_color=BORDER,
                                       command=self.retry)
        self.retry_btn.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row, text="Settings", height=30, fg_color=CARD_HI, hover_color=BORDER,
                      command=self.open_settings).pack(side="left", fill="x",
                                                       expand=True, padx=(8, 0))

        # ---- mode
        self._section(sb, "MODE")
        self.mode_var = ctk.StringVar(value=a["mode"] if a["mode"] in MODES else MODES[0])
        ctk.CTkSegmentedButton(sb, values=MODES, variable=self.mode_var, height=32,
                               selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
                               command=self.set_mode).pack(fill="x", padx=18)

        # ---- scans
        self._section(sb, "IMAGES & SCANS")
        self.scan_list = ctk.CTkScrollableFrame(sb, height=190, fg_color=BG, corner_radius=12,
                                                scrollbar_button_color=BORDER)
        self.scan_list.pack(fill="x", padx=18)
        row = ctk.CTkFrame(sb, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=(8, 0))
        ctk.CTkButton(row, text="Refresh", height=30, fg_color=CARD_HI, hover_color=BORDER,
                      command=self.refresh_scans).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row, text="Open image…", height=30, fg_color=CARD_HI,
                      hover_color=BORDER, command=self.open_image).pack(
            side="left", fill="x", expand=True, padx=(8, 0))
        ctk.CTkButton(row, text="Open folder…", height=30, fg_color=CARD_HI,
                      hover_color=BORDER, command=self.open_folder).pack(
            side="left", fill="x", expand=True, padx=(8, 0))
        self.hint_lbl = ctk.CTkLabel(sb, text="", font=font(11), text_color=MUTED,
                                     justify="left")
        self.hint_lbl.pack(anchor="w", padx=20, pady=(6, 0))
        self.nav_box = ctk.CTkFrame(sb, fg_color="transparent", height=1)
        self.nav_box.pack(fill="x", padx=18)
        self.nav = ctk.CTkFrame(self.nav_box, fg_color=CARD, corner_radius=10)
        self.nav.columnconfigure(1, weight=1)
        ctk.CTkButton(self.nav, text="◀", width=34, height=30, fg_color=CARD_HI,
                      hover_color=BORDER, command=lambda: self.step_frame(-1)).grid(
            row=0, column=0, padx=(6, 4), pady=(6, 2))
        self.nav_lbl = ctk.CTkLabel(self.nav, text="", font=font(12, "bold"))
        self.nav_lbl.grid(row=0, column=1, sticky="ew")
        ctk.CTkButton(self.nav, text="▶", width=34, height=30, fg_color=CARD_HI,
                      hover_color=BORDER, command=lambda: self.step_frame(1)).grid(
            row=0, column=2, padx=(4, 6), pady=(6, 2))
        self.nav_file = ctk.CTkLabel(self.nav, text="", font=font(11), text_color=MUTED)
        self.nav_file.grid(row=1, column=0, columnspan=3)
        self.batch_btn = ctk.CTkButton(self.nav, text="Analyze all images", height=30,
                                       fg_color=ACCENT, hover_color=ACCENT_HOVER,
                                       command=self.analyze_all)
        self.batch_btn.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(2, 6))
        self.auto_var = ctk.BooleanVar(value=a["auto"])
        self.auto_sw = ctk.CTkSwitch(sb, text="", variable=self.auto_var,
                                     progress_color=ACCENT, font=font(12))
        self.auto_sw.pack(anchor="w", padx=18, pady=(10, 0))

        # ---- analysis
        self._section(sb, "ANALYSIS")
        grid = ctk.CTkFrame(sb, fg_color="transparent")
        grid.pack(fill="x", padx=18)
        grid.columnconfigure(1, weight=1)
        ctk.CTkLabel(grid, text="Channel", font=font(12)).grid(row=0, column=0, sticky="w")
        self.channel_var = ctk.StringVar(value=a["channel"])
        ctk.CTkSegmentedButton(grid, values=list(P.CHANNELS), variable=self.channel_var,
                               selected_color=ACCENT, selected_hover_color=ACCENT_HOVER
                               ).grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=3)
        ctk.CTkLabel(grid, text="Rotate", font=font(12)).grid(row=1, column=0, sticky="w")
        self.rotate_var = ctk.StringVar(value=f"{a['rotate']}°")
        ctk.CTkSegmentedButton(grid, values=["0°", "90°", "180°", "270°"],
                               variable=self.rotate_var, selected_color=ACCENT,
                               selected_hover_color=ACCENT_HOVER
                               ).grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=3)

        self.win_var = ctk.IntVar(value=a["win"])
        self._slider(sb, "Window", 3, 21, 9, self.win_var,
                                      lambda v: f"{int(v)}×{int(v)} px", self._on_win)
        self.align_var = ctk.BooleanVar(value=a["align"])
        ctk.CTkCheckBox(sb, text="Auto-align scans (phase correlation)",
                        variable=self.align_var, font=font(12), fg_color=ACCENT,
                        hover_color=ACCENT_HOVER).pack(anchor="w", padx=18, pady=(8, 0))
        self.analyze_btn = ctk.CTkButton(sb, text="Analyze", height=44, corner_radius=12,
                                         font=font(16, "bold"), fg_color=ACCENT,
                                         hover_color=ACCENT_HOVER, command=self.analyze)
        self.analyze_btn.pack(fill="x", padx=18, pady=(14, 0))

        # ---- detection (live)
        self._section(sb, "DETECTION  ·  live")
        self.method_var = ctk.StringVar(value=a["method"])
        self.method_btn = ctk.CTkSegmentedButton(
            sb, values=list(METHODS), variable=self.method_var, selected_color=ACCENT,
            selected_hover_color=ACCENT_HOVER, command=lambda _: self.schedule_eval())
        self.method_btn.pack(fill="x", padx=18)
        self.kthr_var = ctk.DoubleVar(value=a["k_thr"])
        self.kthr_title = self._slider(sb, "Contrast change", 5, 100, 95, self.kthr_var,
                     lambda v: f"> {v:.0f} %", lambda _: self.schedule_eval())
        self.cthr_var = ctk.DoubleVar(value=a["corr_thr"])
        self._slider(sb, "Correlation below", 0.1, 0.95, 85, self.cthr_var,
                     lambda v: f"< {v:.2f}", lambda _: self.schedule_eval())
        self.area_var = ctk.DoubleVar(value=a["min_area"])
        self._slider(sb, "Alarm if area", 0.1, 20, 199, self.area_var,
                     lambda v: f"> {v:.1f} %", lambda _: self.schedule_eval())
        ctk.CTkLabel(sb, text=" ", height=10).pack()

    def _slider(self, parent, title, lo, hi, steps, var, fmt, cb):
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.pack(fill="x", padx=18, pady=(10, 0))
        top = ctk.CTkFrame(box, fg_color="transparent")
        top.pack(fill="x")
        name = ctk.CTkLabel(top, text=title, font=font(12))
        name.pack(side="left")
        val = ctk.CTkLabel(top, text=fmt(var.get()), font=font(12, "bold"), text_color=ACCENT)
        val.pack(side="right")

        def changed(v):
            cb(v)
            val.configure(text=fmt(var.get()))
        ctk.CTkSlider(box, from_=lo, to=hi, number_of_steps=steps, variable=var,
                      progress_color=ACCENT, button_color=ACCENT,
                      button_hover_color=ACCENT_HOVER, command=changed).pack(fill="x", pady=(2, 0))
        return name

    def _on_win(self, v):
        v = int(round(v))
        if v % 2 == 0:
            v += 1
        self.win_var.set(v)

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew", padx=(18, 18), pady=(16, 10))
        main.columnconfigure(0, weight=1)
        main.rowconfigure(3, weight=1)

        # ---- verdict banner
        banner = ctk.CTkFrame(main, fg_color=CARD, corner_radius=14,
                              border_width=1, border_color=BORDER)
        banner.grid(row=0, column=0, sticky="ew")
        banner.columnconfigure(1, weight=1)
        self.stripe = ctk.CTkFrame(banner, width=8, height=10, fg_color=BORDER, corner_radius=4)
        self.stripe.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(12, 14), pady=12)
        self.verdict_lbl = ctk.CTkLabel(banner, text="NO ANALYSIS YET", font=font(26, "bold"),
                                        text_color=MUTED, anchor="w")
        self.verdict_lbl.grid(row=0, column=1, sticky="w", pady=(12, 0))
        self.verdict_sub = ctk.CTkLabel(banner, text="Choose scans on the left",
                                        font=font(13), text_color=MUTED, anchor="w")
        self.verdict_sub.grid(row=1, column=1, sticky="w", pady=(0, 12))
        pair = ctk.CTkFrame(banner, fg_color="transparent")
        pair.grid(row=0, column=2, rowspan=2, padx=16)
        self.ref_lbl = ctk.CTkLabel(pair, text="REF  —", font=font(13, "bold"),
                                    text_color=ACCENT, anchor="e")
        self.ref_lbl.pack(anchor="e")
        self.cmp_lbl = ctk.CTkLabel(pair, text="CMP  frames within REF", font=font(13, "bold"),
                                    text_color=CMP_COLOR, anchor="e")
        self.cmp_lbl.pack(anchor="e")

        # ---- stat cards
        cards = ctk.CTkFrame(main, fg_color="transparent")
        cards.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self.cards = {}
        for i, (key, title) in enumerate((("k", "Speckle contrast K"), ("dk", "ΔK"),
                                          ("area", "Changed area"), ("corr", "Correlation"),
                                          ("sat", "Saturation"))):
            cards.columnconfigure(i, weight=1, uniform="c")
            c = StatCard(cards, title)
            c.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0))
            self.cards[key] = c

        # ---- view bar
        bar = ctk.CTkFrame(main, fg_color="transparent")
        bar.grid(row=2, column=0, sticky="ew", pady=(12, 8))
        self.view_btn = ctk.CTkSegmentedButton(bar, values=VIEWS, command=self.set_view,
                                               selected_color=ACCENT,
                                               selected_hover_color=ACCENT_HOVER,
                                               height=32, font=font(12))
        self.view_btn.set(self.view)
        self.view_btn.pack(side="left")
        ctk.CTkButton(bar, text="Export", width=90, height=32, fg_color=CARD_HI,
                      hover_color=BORDER, command=self.export).pack(side="right")
        ctk.CTkButton(bar, text="Reset zoom", width=100, height=32, fg_color=CARD_HI,
                      hover_color=BORDER, command=self.reset_zoom).pack(side="right", padx=8)

        # ---- canvas
        card = ctk.CTkFrame(main, fg_color=CARD, corner_radius=14,
                            border_width=1, border_color=BORDER)
        card.grid(row=3, column=0, sticky="nsew")
        self.fig = Figure(facecolor=CARD)
        self.canvas = FigureCanvasTkAgg(self.fig, master=card)
        w = self.canvas.get_tk_widget()
        w.configure(bg=CARD, highlightthickness=0)
        w.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", lambda e: setattr(self, "_pan", None))

        # ---- ROI + status
        info = ctk.CTkFrame(main, fg_color="transparent")
        info.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self.roi_lbl = ctk.CTkLabel(info, text="ROI: left-drag on the image",
                                    font=font(12), text_color=TEXT, anchor="w")
        self.roi_lbl.pack(side="left")
        ctk.CTkButton(info, text="Clear ROI", width=80, height=24, fg_color=CARD_HI,
                      hover_color=BORDER, font=font(11),
                      command=self.clear_roi).pack(side="left", padx=10)
        ctk.CTkLabel(info, text="scroll zoom · right-drag pan · double-click reset",
                     font=font(11), text_color=MUTED).pack(side="right")

        status = ctk.CTkFrame(main, fg_color=PANEL, corner_radius=10)
        status.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        status.columnconfigure(1, weight=1)
        self.dot = ctk.CTkLabel(status, text="●", text_color=OK, font=font(14))
        self.dot.grid(row=0, column=0, padx=(12, 6), pady=6)
        self.status_lbl = ctk.CTkLabel(status, text="Ready", font=font(12),
                                       text_color=TEXT, anchor="w")
        self.status_lbl.grid(row=0, column=1, sticky="ew")
        self.progress = ctk.CTkProgressBar(status, width=180, progress_color=ACCENT,
                                           fg_color=BORDER)
        self.progress.set(0)
        self.progress.grid(row=0, column=2, padx=12)

    # ============================================================ threading
    def _ui(self, fn, *args):
        """Queue a call to run on the Tk thread (worker threads use this)."""
        self.q.put((fn, args))

    def _pump(self):
        try:
            while True:
                fn, args = self.q.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        self.after(50, self._pump)

    def run_bg(self, work, done=None):
        if self.busy:
            self.set_status("Busy - wait for the current job to finish", "warn")
            return
        self.set_busy(True)

        def runner():
            try:
                res = work()
                if done:
                    self._ui(done, res)
            except Exception as e:  # noqa: BLE001 - show any failure in the status bar
                self._ui(self.set_status, f"Error: {e}", "bad")
            finally:
                self._ui(self.set_busy, False)
        threading.Thread(target=runner, daemon=True).start()

    def set_busy(self, busy):
        self.busy = busy
        st = "disabled" if busy else "normal"
        self.batch_btn.configure(state=st)
        self.scan_btn.configure(state=st)
        self.analyze_btn.configure(state=st)
        self.retry_btn.configure(state="normal" if (self.capture.pending and not busy)
                                 else "disabled")
        if busy:
            self.dot.configure(text_color=WARN)
        else:
            self.progress.set(0)

    def set_status(self, msg, level="info"):
        col = {"info": TEXT, "ok": OK, "warn": WARN, "bad": BAD}[level]
        self.status_lbl.configure(text=msg, text_color=col)
        self.dot.configure(text_color={"info": OK, "ok": OK, "warn": WARN, "bad": BAD}[level])
        print(msg)

    def set_progress(self, msg, frac):
        self.status_lbl.configure(text=msg, text_color=TEXT)
        self.progress.set(frac)

    # ============================================================ scans
    def _scan_dirs(self) -> list[Path]:
        inc = Path(self.settings["incoming"])
        if not inc.is_dir():
            return []
        return [d for d in inc.iterdir() if d.is_dir()]

    def refresh_scans(self):
        for w in self.scan_list.winfo_children():
            w.destroy()
        dirs = sorted(self._scan_dirs(), key=lambda d: d.stat().st_mtime, reverse=True)
        items = [(d, False) for d in dirs] + [(d, True) for d in self.extra_dirs if d.exists()]
        if not items:
            ctk.CTkLabel(self.scan_list, text=f"No scans in\n{self.settings['incoming']}",
                         text_color=MUTED, font=font(12)).pack(pady=20)
        for d, ext in items:
            try:
                n = len(P.list_frames(d))
            except OSError:
                continue
            done = isinstance(d, Path) and d.is_dir() and (d / "done.txt").exists()
            ScanRow(self.scan_list, self, d, n, done, ext).pack(fill="x", pady=3, padx=2)
        self._pair_labels()
        self._update_nav()

    @property
    def single(self) -> bool:
        return self.mode_var.get() == MODES[0]

    def _pair_labels(self, c=None):
        if self.single:
            self.ref_lbl.configure(text=f"IMAGE  {c.label_a if c else (self.ref.name if self.ref else '—')}")
            self.cmp_lbl.configure(text="no reference needed")
        else:
            self.ref_lbl.configure(text=f"REF  {c.label_a if c else (self.ref.name if self.ref else '—')}")
            self.cmp_lbl.configure(text=f"CMP  {c.label_b if c else (self.cmp.name if self.cmp else 'frames within REF')}")

    def set_mode(self, mode):
        single = mode == MODES[0]
        self.view_btn.configure(values=SINGLE_VIEWS if single else VIEWS)
        self.view = "Overlay"
        self.view_btn.set(self.view)
        self.method_btn.configure(state="disabled" if single else "normal")
        self.kthr_title.configure(text="K below surroundings" if single else "Contrast change")
        self.hint_lbl.configure(text=(
            "Press USE on an image, folder or image group.\nStep with ◀ ▶ or analyse them all."
            if single else "No CMP picked → frame 1 of REF is compared\nwith its other frames."))
        self.auto_sw.configure(text="Auto-analyse every new scan" if single
                               else "Auto-compare new scans with REF")
        self.comp = self.verdict = None
        self.ax = None
        self.roi = None
        self._lims = None
        self.stripe.configure(fg_color=BORDER)
        self.verdict_lbl.configure(text="NO ANALYSIS YET", text_color=MUTED)
        self.verdict_sub.configure(text="Pick an image on the left and press Analyze" if single
                                   else "Pick REF (and optionally CMP) scans and press Analyze")
        for card in self.cards.values():
            card.set("—")
        self.refresh_scans()
        self._update_roi_label()
        self._render_placeholder(
            "Pick ONE image (USE), then press Analyze" if single
            else "Pick a REF scan (and optionally a CMP scan), then press Analyze")

    def pick(self, path, role):
        if role == "ref":
            if path != self.ref:
                self.frame_idx = 0
            self.ref = path
            if self.cmp == path:
                self.cmp = None
        else:
            self.cmp = None if (path is None or path == self.cmp) else path
            if self.cmp == self.ref:
                self.ref = None
        self.refresh_scans()

    def open_image(self):
        fs = filedialog.askopenfilenames(
            title="Speckle image(s) - Ctrl/Shift-click to pick several",
            filetypes=[("Speckle images", "*.dng *.DNG *.png *.PNG *.tif *.tiff *.TIF "
                                          "*.jpg *.jpeg *.JPG *.bmp"),
                       ("All files", "*")])
        if fs:
            files = tuple(sorted(Path(f) for f in fs))
            p = files[0] if len(files) == 1 else P.FileSet(files)
            self.frame_idx = 0
            if p not in self.extra_dirs:
                self.extra_dirs.append(p)
            self.ref = p
            if self.cmp == p:
                self.cmp = None
            self.refresh_scans()

    def _item_files(self):
        if self.ref is None:
            return []
        try:
            return P.list_frames(self.ref)
        except OSError:
            return []

    def _update_nav(self):
        files = self._item_files() if self.single else []
        if len(files) > 1:
            self.frame_idx = min(self.frame_idx, len(files) - 1)
            self.nav_lbl.configure(text=f"Image {self.frame_idx + 1} / {len(files)}")
            self.nav_file.configure(text=files[self.frame_idx].name)
            if not self.nav.winfo_ismapped():
                self.nav.pack(fill="x", pady=(8, 0))
        else:
            self.nav.pack_forget()

    def step_frame(self, d):
        files = self._item_files()
        if len(files) < 2:
            return
        self.show_frame((self.frame_idx + d) % len(files))

    def show_frame(self, i):
        self.frame_idx = i
        self._update_nav()
        if not self.busy:
            self.analyze()

    def _arrow(self, e, d):
        if self.single and "entry" not in str(e.widget).lower():
            self.step_frame(d)

    def open_folder(self):
        d = filedialog.askdirectory(title="Folder with speckle frames (DNG/PNG/TIFF/JPG)")
        if d:
            p = Path(d)
            if p not in self.extra_dirs:
                self.extra_dirs.append(p)
            if not self.ref or self.single:
                self.ref = p
                self.frame_idx = 0
            self.refresh_scans()

    def _watch(self):
        """Poll incoming/ for new scan folders (e.g. from another laptop or the
        standalone capture script) and auto-compare finished ones."""
        try:
            dirs = set(self._scan_dirs())
            new = dirs - self.known
            self.known |= new
            self.waiting |= new
            ready = {d for d in self.waiting if (d / "done.txt").exists()}
            self.waiting -= ready
            self.waiting &= dirs
            if new or ready:
                self.refresh_scans()
            for d in sorted(ready):
                self._auto_compare(d)
        finally:
            self.after(2000, self._watch)

    def _auto_compare(self, d: Path):
        if not self.auto_var.get():
            return
        if self.single:
            if not self.busy:
                self.ref = d
                self.frame_idx = 0
                self.refresh_scans()
                self.analyze()
            return
        if self.ref is None:
            self.ref = d
            self.set_status(f"New scan {d.name} set as REF", "ok")
            self.refresh_scans()
        elif d != self.ref and not self.busy:
            self.cmp = d
            self.refresh_scans()
            self.analyze()

    # ============================================================ capture
    def _sync_capture_cfg(self):
        self.settings["capture"]["esp_ip"] = self.ip_var.get().strip()
        self.capture.cfg.update(self.settings["capture"])
        self.capture.incoming = self.settings["incoming"]

    def scan(self):
        self._sync_capture_cfg()
        self.run_bg(self.capture.scan, self._scan_done)

    def retry(self):
        self._sync_capture_cfg()
        self.run_bg(self.capture.retry, self._scan_done)

    def _scan_done(self, folder):
        if not folder:
            self.set_status(self.status_lbl.cget("text"), "warn")
            return
        d = Path(folder)
        self.known.add(d)
        self.waiting.discard(d)
        self.refresh_scans()
        if self.capture.pending:
            self.set_status(self.status_lbl.cget("text"), "warn")
            return
        self.set_status(self.status_lbl.cget("text"), "ok")
        # run after set_busy(False) has been processed
        self.after(100, lambda: self._auto_compare(d))

    def detect_ip(self):
        self.detect_btn.configure(state="disabled", text="…")
        self.set_status("Reading ESP32 IP over USB serial (close the Arduino serial monitor)...")

        def work():
            ip = detect_esp_ip()

            def done():
                self.detect_btn.configure(state="normal", text="Detect")
                if ip:
                    self.ip_var.set(ip)
                    self.set_status(f"ESP32 found at {ip}", "ok")
                else:
                    self.set_status("Could not detect the ESP32 IP - type it in instead", "warn")
            self._ui(done)
        threading.Thread(target=work, daemon=True).start()

    def open_settings(self):
        SettingsDialog(self)

    # ============================================================ analysis
    def _analysis_settings(self) -> P.Settings:
        return P.Settings(channel=self.channel_var.get(),
                          rotate=int(self.rotate_var.get().rstrip("°")),
                          win=int(self.win_var.get()) | 1,
                          align=self.align_var.get())

    def _load(self, path, channel, rotate, warns):
        key = (str(path), channel, rotate)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        frames = P.load_scan(path, channel, rotate, warn=warns.append)
        if frames:
            self.cache[key] = frames
            while len(self.cache) > 4:
                self.cache.popitem(last=False)
        return frames

    @staticmethod
    def _why(warns):
        """Summarise why frames failed: the first reason plus how many failed."""
        if not warns:
            return "The folder has no .dng/.png/.tif/.jpg files"
        more = f" (and {len(warns) - 1} more like it)" if len(warns) > 1 else ""
        return warns[0] + more

    def analyze(self):
        if self.busy:
            return
        if not self.ref:
            self.set_status("Pick an image first (USE button / Open image…)" if self.single
                            else "Pick a reference scan first (REF button)", "warn")
            return
        s = self._analysis_settings()
        ref, cmp, single = self.ref, self.cmp, self.single
        if single:
            files = self._item_files()
            if not files:
                self.set_status(f"No images in {ref.name}", "bad")
                return
            idx = min(self.frame_idx, len(files) - 1)

        def work():
            warns: list[str] = []

            def prog(msg, f):
                self._ui(self.set_progress, msg, f)
            if single:
                f = files[idx]
                prog(f"Loading {f.name}...", 0.05)
                try:
                    frame = P.load_frame(f, s.channel, s.rotate)
                except Exception as e:  # noqa: BLE001
                    raise ValueError(f"Cannot read {f.name}: {e}") from e
                label = f.name if len(files) == 1 else f"{f.name} ({idx + 1}/{len(files)})"
                return P.analyze_single(frame, s, label, prog)
            prog(f"Loading {ref.name}...", 0.05)
            fa = self._load(ref, s.channel, s.rotate, warns)
            if not fa:
                raise ValueError(f"No readable frames in {ref.name}. {self._why(warns)}")
            if cmp:
                prog(f"Loading {cmp.name}...", 0.15)
                fb = self._load(cmp, fa[0].channel, s.rotate, warns)
                if not fb:
                    raise ValueError(f"No readable frames in {cmp.name}. {self._why(warns)}")
                c = P.compare(fa, fb, s, ref.name, cmp.name, prog)
            else:
                if len(fa) < 2:
                    raise ValueError(f"{ref.name} has only one readable frame - pick a CMP scan")
                c = P.compare(fa[:1], fa[1:], s, f"{ref.name} · frame 1",
                              f"{ref.name} · frame{'s 2–' + str(len(fa)) if len(fa) > 2 else ' 2'}", prog, temporal_frames=fa)
            c.warnings = warns + c.warnings
            return c
        self.run_bg(work, self._analysis_done)

    def analyze_all(self):
        """Analyse every image of the selected folder / image group on its own."""
        files = self._item_files()
        if self.busy or not files:
            return
        s = self._analysis_settings()
        thr, area = self.kthr_var.get() / 100, self.area_var.get()
        name = self.ref.name

        def work():
            rows = []
            for i, f in enumerate(files):
                self._ui(self.set_progress, f"Analysing {f.name} ({i + 1}/{len(files)})...",
                         (i + 1) / len(files))
                try:
                    c = P.analyze_single(P.load_frame(f, s.channel, s.rotate), s, f.name)
                    v = P.evaluate(c, "contrast", thr, 0.5, area)
                    rows.append(dict(index=i, image=f.name, flow=v.changed,
                                     flow_area_pct=round(v.changed_pct, 2),
                                     still_K=round(v.mean_k_a, 4),
                                     flow_K=round(v.mean_k_b, 4) if v.changed_pct else None,
                                     saturation_pct=round(c.sat_a * 100, 3), error=""))
                except Exception as e:  # noqa: BLE001 - keep going, report per image
                    rows.append(dict(index=i, image=f.name, flow=None, flow_area_pct=None,
                                     still_K=None, flow_K=None, saturation_pct=None,
                                     error=str(e)))
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
            out = ROOT / "results" / f"{safe}_batch_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            return rows, out

        def done(res):
            rows, out = res
            BatchWindow(self, name, rows, out, thr * 100)
            n_flow = sum(1 for r in rows if r["flow"])
            self.set_status(f"Analysed {len(rows)} images · flow in {n_flow} · saved {out.name}",
                            "ok")
        self.run_bg(work, done)

    def _analysis_done(self, c: P.Comparison):
        old = self.comp
        self.comp = c
        if old is None or old.k_a.shape != c.k_a.shape:
            self.roi = None
            self._lims = None
            self.ax = None
        self._pair_labels(c)
        self.evaluate(render=False)
        self.render()
        what = "1 image" if c.single else f"{c.n_a}+{c.n_b} frames"
        msg = (f"Analyzed {what} · channel {c.channel} · "
               f"{c.k_a.shape[1]}×{c.k_a.shape[0]} px")
        if c.shift != (0, 0):
            msg += f" · aligned by dx={c.shift[1]}, dy={c.shift[0]} px"
        if c.warnings:
            self.set_status(msg + "  ⚠ " + "; ".join(c.warnings), "warn")
        else:
            self.set_status(msg, "ok")

    def schedule_eval(self):
        if self._eval_job:
            self.after_cancel(self._eval_job)
        self._eval_job = self.after(150, self.evaluate)

    def evaluate(self, render=True):
        self._eval_job = None
        c = self.comp
        if c is None:
            return
        v = P.evaluate(c, METHODS[self.method_var.get()], self.kthr_var.get() / 100,
                       self.cthr_var.get(), self.area_var.get())
        self.verdict = v
        color = BAD if v.changed else OK
        if c.single:
            self._show_single_verdict(c, v, color)
            if render and self.view == "Overlay":
                self.render()
            self._update_roi_label()
            return
        self.stripe.configure(fg_color=color)
        self.verdict_lbl.configure(text="CHANGE DETECTED" if v.changed else "NO SIGNIFICANT CHANGE",
                                   text_color=color)
        self.verdict_sub.configure(
            text=f"{v.changed_pct:.1f}% of the lit area changed (alarm above "
                 f"{self.area_var.get():.1f}%) · {self.method_var.get().lower()} method · "
                 f"{v.valid_pct:.0f}% of the image has enough laser light")
        self.cards["k"].set(f"{v.mean_k_a:.3f} → {v.mean_k_b:.3f}", "mean over lit area",
                            title="Speckle contrast K")
        self.cards["dk"].set(f"{v.dk_pct:+.1f} %",
                             "lower K = more motion" if v.dk_pct < 0 else "higher K = less motion",
                             BAD if abs(v.dk_pct) > self.kthr_var.get() else TEXT, title="ΔK")
        self.cards["area"].set(f"{v.changed_pct:.1f} %", "of lit pixels", color,
                               title="Changed area")
        self.cards["corr"].set(f"{v.mean_corr:.2f}", "1 = identical pattern",
                               WARN if v.mean_corr < self.cthr_var.get() else TEXT,
                               title="Correlation")
        sat = max(c.sat_a, c.sat_b) * 100
        self.cards["sat"].set(f"{sat:.2f} %", "ref {:.2f} · cmp {:.2f}".format(
            c.sat_a * 100, c.sat_b * 100), WARN if sat > 0.5 else TEXT)
        if render and self.view == "Overlay":
            self.render()
        self._update_roi_label()

    def _show_single_verdict(self, c, v, color):
        thr = self.kthr_var.get()
        self.stripe.configure(fg_color=color)
        self.verdict_lbl.configure(text="FLOW DETECTED" if v.changed else "NO FLOW REGIONS",
                                   text_color=color)
        self.verdict_sub.configure(
            text=f"{v.changed_pct:.1f}% of the lit area has speckle contrast ≥ {thr:.0f}% below "
                 f"its surroundings (alarm above {self.area_var.get():.1f}%) · "
                 f"{v.valid_pct:.0f}% of the image has enough laser light")
        flagged = v.changed_pct > 0 and np.isfinite(v.mean_k_b)
        self.cards["k"].set(f"{v.mean_k_a:.3f}", "mean K of still areas", title="Still areas K")
        self.cards["dk"].set(f"{v.mean_k_b:.3f}" if flagged else "—",
                             f"{v.dk_pct:+.0f}% vs surroundings" if flagged else "nothing flagged",
                             BAD if flagged else TEXT, title="Flow areas K")
        self.cards["area"].set(f"{v.changed_pct:.1f} %", "of lit pixels", color,
                               title="Flow area")
        flow = (v.mean_k_a / v.mean_k_b) ** 2 if flagged else float("nan")
        self.cards["corr"].set(f"×{flow:.1f}" if flagged else "—",
                               "flow index 1/K² vs still", title="Relative flow")
        sat = c.sat_a * 100
        self.cards["sat"].set(f"{sat:.2f} %", "pixels at sensor maximum",
                              WARN if sat > 0.5 else TEXT, title="Saturation")

    # ============================================================ rendering
    def set_view(self, v):
        self.view = v
        self.render()

    def _stride(self, a):
        return max(1, int(np.ceil(max(a.shape) / MAX_DISPLAY_PX)))

    def _view_data(self):
        c, v = self.comp, VIEW_ALIAS.get(self.view, self.view)
        if c.single:
            return self._single_view_data(c, v)
        st = self._stride(c.k_a)
        ds = lambda a: a[::st, ::st]  # noqa: E731
        valid = ds(c.valid)
        nan = lambda a: np.where(valid, a, np.nan)  # noqa: E731
        bright = max(np.percentile(ds(c.intensity_a), 99.5), 1.0)
        gray = lambda a: np.clip(np.sqrt(np.clip(a, 0, None) / bright), 0, 1)  # noqa: E731

        if v == "Overlay":
            base = gray(ds(c.intensity_a))
            base = np.where(valid, base, base * 0.45)
            rgb = np.repeat(base[..., None], 3, axis=2)
            if self.verdict is not None:
                m = ds(self.verdict.mask)
                red = np.array([1.0, 0.3, 0.42])
                rgb[m] = rgb[m] * 0.35 + red * 0.65
            return dict(img=rgb, title=f"Changed regions (red) on {c.label_a}")
        if v in ("Reference", "Compare"):
            a = c.intensity_a if v == "Reference" else c.intensity_b
            lab = c.label_a if v == "Reference" else c.label_b
            return dict(img=gray(ds(a)), cmap="gray", vmin=0, vmax=1,
                        title=f"Mean intensity · {lab}", cbar="√ intensity (normalised)")
        if v in ("K ref", "K cmp", "Temporal K"):
            if v == "Temporal K":
                if c.temporal_k is None:
                    return dict(msg="Temporal contrast needs ≥ 2 frames in REF")
                a, lab = c.temporal_k, f"Temporal contrast · {c.label_a.split(' · ')[0]}"
            else:
                a = c.k_a if v == "K ref" else c.k_b
                lab = f"Spatial contrast K · {c.label_a if v == 'K ref' else c.label_b}"
            vals = np.concatenate([ds(c.k_a)[valid], ds(c.k_b)[valid]])
            hi = max(float(np.percentile(vals, 99.5)) if vals.size else 1.0, 0.05)
            return dict(img=nan(ds(a)), cmap="viridis", vmin=0, vmax=hi, title=lab, cbar="K")
        if v == "ΔK/K":
            d = ds(c.dk_rel) * 100
            lim = max(2 * self.kthr_var.get(),
                      float(np.percentile(np.abs(d[valid]), 99)) if valid.any() else 50)
            lim = min(lim, 200)
            return dict(img=nan(d), cmap="RdBu_r", vmin=-lim, vmax=lim,
                        title="Relative change in speckle contrast (cmp vs ref)",
                        cbar="ΔK / K  [%]")
        if v == "Correlation":
            return dict(img=nan(ds(c.corr)), cmap="magma", vmin=0, vmax=1,
                        title="Speckle pattern correlation (dark = pattern changed)",
                        cbar="correlation")
        if v == "Flow ratio":
            r = 2 * np.log2((ds(c.k_a) + 1e-9) / (ds(c.k_b) + 1e-9))
            return dict(img=nan(r), cmap="PuOr_r", vmin=-2, vmax=2,
                        title="Relative flow index 1/K² (cmp ÷ ref)",
                        cbar="log₂ ratio  (+1 = 2× more flow)")
        return dict(msg="?")

    def _single_view_data(self, c, v):
        st = self._stride(c.k_a)
        ds = lambda a: a[::st, ::st]  # noqa: E731
        valid = ds(c.valid)
        nan = lambda a: np.where(valid, a, np.nan)  # noqa: E731
        bright = max(np.percentile(ds(c.intensity_a), 99.5), 1.0)
        gray = np.clip(np.sqrt(np.clip(ds(c.intensity_a), 0, None) / bright), 0, 1)
        if v == "Overlay":
            base = np.where(valid, gray, gray * 0.45)
            rgb = np.repeat(base[..., None], 3, axis=2)
            if self.verdict is not None:
                m = ds(self.verdict.mask)
                rgb[m] = rgb[m] * 0.35 + np.array([1.0, 0.3, 0.42]) * 0.65
            return dict(img=rgb, title=f"Likely flow / motion (red) · {c.label_a}")
        if v == "Reference":
            return dict(img=gray, cmap="gray", vmin=0, vmax=1, title=f"Image · {c.label_a}",
                        cbar="√ intensity (normalised)")
        if v in ("K ref", "K cmp"):
            vals = ds(c.k_a)[valid]
            hi = max(float(np.percentile(vals, 99.5)) if vals.size else 1.0, 0.05)
            a = c.k_a if v == "K ref" else c.k_b
            t = ("Speckle contrast K (dark = flow / motion)" if v == "K ref"
                 else "Local surroundings K (what each pixel is compared with)")
            return dict(img=nan(ds(a)), cmap="viridis", vmin=0, vmax=hi, title=t, cbar="K")
        if v == "ΔK/K":
            d = ds(c.dk_rel) * 100
            lim = min(max(2 * self.kthr_var.get(), 20), 100)
            return dict(img=nan(d), cmap="RdBu_r", vmin=-lim, vmax=lim,
                        title="Speckle contrast relative to surroundings (blue = lower)",
                        cbar="K / K_surroundings − 1  [%]")
        if v == "Flow ratio":
            r = -2 * np.log2(np.clip(1 + ds(c.dk_rel), 1e-3, None))
            return dict(img=nan(r), cmap="inferno", vmin=0, vmax=1.5,
                        title="Relative flow index vs surroundings (bright = more flow)",
                        cbar="log₂ (1/K²  ÷  surroundings)")
        return dict(msg="Not used in single-image mode")

    def _style_ax(self, ax):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(BORDER)

    def _render_placeholder(self, text):
        self.fig.clear()
        self.ax = None
        self.selector = None
        ax = self.fig.add_subplot()
        ax.set_facecolor(CARD)
        ax.axis("off")
        ax.text(0.5, 0.55, "◎", ha="center", va="center", fontsize=60, color=BORDER,
                transform=ax.transAxes)
        ax.text(0.5, 0.40, text, ha="center", va="center", fontsize=12, color=MUTED,
                transform=ax.transAxes)
        self.canvas.draw_idle()

    def render(self):
        if self.comp is None:
            return
        d = self._view_data()
        if "msg" in d:
            self._render_placeholder(d["msg"])
            return
        lims = self._lims
        if self.ax is not None:
            lims = (self.ax.get_xlim(), self.ax.get_ylim())
        self.fig.clear()
        h, w = self.comp.k_a.shape
        self.fig.subplots_adjust(left=0.05, right=0.9, top=0.93, bottom=0.06)
        ax = self.fig.add_subplot()
        self._style_ax(ax)
        kw = {} if d.get("cmap") is None else dict(cmap=colormaps[d["cmap"]].with_extremes(
            bad=CARD), vmin=d["vmin"], vmax=d["vmax"])
        im = ax.imshow(d["img"], extent=(0, w, h, 0), interpolation="nearest", **kw)
        ax.set_title(d["title"], color=TEXT, fontsize=11, loc="left", pad=8)
        if d.get("cbar"):
            cb = self.fig.colorbar(im, cax=ax.inset_axes([1.02, 0, 0.025, 1]))
            cb.set_label(d["cbar"], color=MUTED, fontsize=9)
            cb.ax.tick_params(colors=MUTED, labelsize=8)
            cb.outline.set_edgecolor(BORDER)
        if lims:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
        self.ax = ax
        self.selector = RectangleSelector(
            ax, self._on_roi, useblit=True, button=[1], minspanx=4, minspany=4,
            spancoords="data", interactive=True,
            props=dict(edgecolor=ACCENT, facecolor=ACCENT, alpha=0.18, fill=True,
                       linewidth=1.5))
        if self.roi:
            self.selector.extents = self.roi
            self.selector.set_visible(True)
        self.canvas.draw_idle()

    # ============================================================ zoom / pan / ROI
    def _full(self):
        h, w = self.comp.k_a.shape
        return (0, w), (h, 0)

    def reset_zoom(self):
        self._lims = None
        if self.ax is not None and self.comp is not None:
            xl, yl = self._full()
            self.ax.set_xlim(xl)
            self.ax.set_ylim(yl)
            self.canvas.draw_idle()

    def _on_scroll(self, e):
        if self.ax is None or e.inaxes is not self.ax or e.xdata is None:
            return
        f = 1 / 1.25 if e.button == "up" else 1.25
        (fx0, fx1), (fy0, fy1) = self._full()
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        nx = [e.xdata - (e.xdata - x0) * f, e.xdata + (x1 - e.xdata) * f]
        ny = [e.ydata - (e.ydata - y0) * f, e.ydata + (y1 - e.ydata) * f]
        if abs(nx[1] - nx[0]) >= fx1 - fx0:
            nx, ny = [fx0, fx1], [fy0, fy1]
        else:
            nx = self._clamp(nx, fx0, fx1)
            ny = self._clamp(ny, fy1, fy0)
        self.ax.set_xlim(nx)
        self.ax.set_ylim(ny)
        self.canvas.draw_idle()

    @staticmethod
    def _clamp(pair, lo, hi):
        a, b = pair
        flip = a > b
        a, b = min(a, b), max(a, b)
        span = b - a
        if a < lo:
            a, b = lo, lo + span
        if b > hi:
            a, b = hi - span, hi
        return [b, a] if flip else [a, b]

    def _on_press(self, e):
        if self.ax is None or e.inaxes is not self.ax:
            return
        if e.dblclick and e.button == 1:
            self.reset_zoom()
        elif e.button == 3:
            self._pan = (e.x, e.y, self.ax.get_xlim(), self.ax.get_ylim())

    def _on_motion(self, e):
        if not self._pan or e.x is None or self.ax is None:
            return
        px, py, (x0, x1), (y0, y1) = self._pan
        bb = self.ax.bbox
        dx = (e.x - px) * (x1 - x0) / bb.width
        dy = (e.y - py) * (y1 - y0) / bb.height
        (fx0, fx1), (fy0, fy1) = self._full()
        self.ax.set_xlim(self._clamp([x0 - dx, x1 - dx], fx0, fx1))
        self.ax.set_ylim(self._clamp([y0 - dy, y1 - dy], fy1, fy0))
        self.canvas.draw_idle()

    def _on_roi(self, click, release):
        if self.comp is None:
            return
        h, w = self.comp.k_a.shape
        x0, x1 = sorted((click.xdata, release.xdata))
        y0, y1 = sorted((click.ydata, release.ydata))
        self.roi = (max(0, x0), min(w, x1), max(0, y0), min(h, y1))
        self._update_roi_label()

    def clear_roi(self):
        self.roi = None
        if self.selector is not None:
            self.selector.set_visible(False)
            self.canvas.draw_idle()
        self._update_roi_label()

    def _update_roi_label(self):
        if not self.roi or self.comp is None:
            self.roi_lbl.configure(text="ROI: left-drag on the image to measure a region")
            return
        x0, x1, y0, y1 = (int(round(v)) for v in self.roi)
        s = P.roi_stats(self.comp, y0, y1, x0, x1)
        if not s:
            self.roi_lbl.configure(text=f"ROI {x1 - x0}×{y1 - y0} px: not enough laser light")
            return
        if self.comp.single:
            self.roi_lbl.configure(
                text=f"ROI {x1 - x0}×{y1 - y0} px  ·  K {s['K_ref']:.3f}  ·  surroundings "
                     f"{s['K_cmp']:.3f} ({-s['dK_pct'] / (1 + s['dK_pct'] / 100):+.1f}%)")
            return
        self.roi_lbl.configure(
            text=f"ROI {x1 - x0}×{y1 - y0} px  ·  K {s['K_ref']:.3f} → {s['K_cmp']:.3f} "
                 f"({s['dK_pct']:+.1f}%)  ·  corr {s['corr']:.2f}  ·  "
                 f"flow ×{s['flow_ratio']:.2f}")

    # ============================================================ export
    def export(self):
        c, v = self.comp, self.verdict
        if c is None or v is None:
            self.set_status("Nothing to export yet - run an analysis first", "warn")
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe = lambda s: "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)  # noqa
        tag = f"{safe(c.label_a)}_single" if c.single else f"{safe(c.label_a)}_vs_{safe(c.label_b)}"
        out = ROOT / "results" / f"{tag}_{stamp}"
        prev_view = self.view
        views = []
        names = (("Overlay", "Image", "K map", "Surroundings K", "ΔK/K", "Flow") if c.single
                 else ("Overlay", "Reference", "Compare", "K ref", "ΔK/K", "Correlation"))
        for name in names:
            self.view = name
            views.append((name, self._view_data()))
        self.view = prev_view
        summary = {
            "mode": "single image" if c.single else "compare",
            "reference": c.label_a, "compare": None if c.single else c.label_b,
            "channel": c.channel,
            "verdict": ("FLOW DETECTED" if v.changed else "NO FLOW REGIONS") if c.single
            else ("CHANGE DETECTED" if v.changed else "NO SIGNIFICANT CHANGE"),
            "changed_area_pct": round(v.changed_pct, 3),
            "mean_K_ref": v.mean_k_a, "mean_K_cmp": v.mean_k_b,
            "dK_pct": round(v.dk_pct, 3), "mean_correlation": v.mean_corr,
            "lit_area_pct": round(v.valid_pct, 2),
            "saturation_pct": {"ref": c.sat_a * 100, "cmp": c.sat_b * 100},
            "shift_px": {"dy": c.shift[0], "dx": c.shift[1]},
            "settings": {**vars(self._analysis_settings()),
                         "method": self.method_var.get(),
                         "contrast_threshold_pct": self.kthr_var.get(),
                         "correlation_threshold": self.cthr_var.get(),
                         "min_area_pct": self.area_var.get()},
            "roi": self.roi and {"x0": self.roi[0], "x1": self.roi[1],
                                 "y0": self.roi[2], "y1": self.roi[3],
                                 **P.roi_stats(c, *(int(round(t)) for t in
                                                    (self.roi[2], self.roi[3],
                                                     self.roi[0], self.roi[1])))},
            "warnings": c.warnings,
        }
        h, w = c.k_a.shape

        def work():
            out.mkdir(parents=True, exist_ok=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=2))
            fig = Figure(figsize=(16, 9.5), facecolor="white", layout="constrained")
            fig.suptitle(f"{summary['verdict']} - {c.label_a}"
                         f"{'' if c.single else ' vs ' + c.label_b} · "
                         f"{v.changed_pct:.1f}% changed", fontsize=14)
            for i, (name, d) in enumerate(views):
                ax = fig.add_subplot(2, 3, i + 1)
                ax.set_title(d["title"], fontsize=9)
                ax.axis("off")
                if "img" in d:
                    kw = {} if d.get("cmap") is None else dict(
                        cmap=d["cmap"], vmin=d["vmin"], vmax=d["vmax"])
                    im = ax.imshow(d["img"], extent=(0, w, h, 0), **kw)
                    if d.get("cbar"):
                        fig.colorbar(im, ax=ax, fraction=0.04).set_label(d["cbar"], fontsize=8)
            fig.savefig(out / "overview.png", dpi=130)
            from PIL import Image
            Image.fromarray((v.mask * 255).astype(np.uint8)).save(out / "change_mask.png")
            np.savez_compressed(out / "maps.npz", K_ref=c.k_a.astype(np.float32),
                                K_cmp=c.k_b.astype(np.float32),
                                dK_rel=c.dk_rel.astype(np.float32),
                                corr=c.corr.astype(np.float32), valid=c.valid, mask=v.mask)
            return out
        self.run_bg(work, lambda p: self.set_status(f"Exported to {p}", "ok"))

    # ============================================================ shutdown
    def on_close(self):
        self.settings["capture"]["esp_ip"] = self.ip_var.get().strip()
        self.settings["analysis"].update(
            channel=self.channel_var.get(), rotate=int(self.rotate_var.get().rstrip("°")),
            win=int(self.win_var.get()) | 1, align=self.align_var.get(),
            method=self.method_var.get(), k_thr=self.kthr_var.get(),
            corr_thr=self.cthr_var.get(), min_area=self.area_var.get(),
            auto=self.auto_var.get(), mode=self.mode_var.get())
        save_settings(self.settings)
        self.destroy()


def save_settings(s):
    try:
        SETTINGS_FILE.write_text(json.dumps(s, indent=2))
    except OSError as e:
        print(f"Could not save settings: {e}")


class BatchWindow(ctk.CTkToplevel):
    """Results table of 'Analyze all images'. Click a row to open that image."""
    COLS = [("#", 40), ("Image", 210), ("Verdict", 130), ("Flow area", 90),
            ("Still K", 80), ("Flow K", 80), ("", 70)]

    def __init__(self, app: App, name, rows, csv_path, thr):
        super().__init__(app, fg_color=PANEL)
        self.title(f"Batch · {name}")
        self.geometry("780x620")
        self.transient(app)
        ok_rows = [r for r in rows if r["error"] == ""]
        n_flow = sum(1 for r in ok_rows if r["flow"])
        mean_area = (sum(r["flow_area_pct"] for r in ok_rows) / len(ok_rows)) if ok_rows else 0
        ctk.CTkLabel(self, text=name, font=font(18, "bold")).pack(anchor="w", padx=18, pady=(16, 0))
        ctk.CTkLabel(self, text=f"Flow detected in {n_flow} / {len(rows)} images  ·  mean flow "
                                f"area {mean_area:.1f}%  ·  threshold K {thr:.0f}% below "
                                f"surroundings", font=font(12), text_color=MUTED).pack(
            anchor="w", padx=18)
        ctk.CTkLabel(self, text=f"Saved: {csv_path}", font=font(11), text_color=MUTED).pack(
            anchor="w", padx=18, pady=(0, 8))
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=22)
        for i, (t, w) in enumerate(self.COLS):
            ctk.CTkLabel(head, text=t.upper(), width=w, anchor="w", font=font(10, "bold"),
                         text_color=MUTED).grid(row=0, column=i)
        body = ctk.CTkScrollableFrame(self, fg_color=BG, corner_radius=12)
        body.pack(fill="both", expand=True, padx=18, pady=(4, 18))
        top = max((r["flow_area_pct"] or 0 for r in ok_rows), default=0)
        for r in rows:
            line = ctk.CTkFrame(body, fg_color=CARD, corner_radius=8)
            line.pack(fill="x", pady=2)
            if r["error"]:
                vals = [str(r["index"] + 1), r["image"], ("error", BAD), r["error"][:40], "", ""]
            else:
                hot = r["flow"] and r["flow_area_pct"] == top
                verdict = ("FLOW" + ("  ▲ max" if hot else ""), BAD) if r["flow"] else ("no flow", OK)
                vals = [str(r["index"] + 1), r["image"], verdict, f"{r['flow_area_pct']:.1f} %",
                        f"{r['still_K']:.3f}",
                        f"{r['flow_K']:.3f}" if r["flow_K"] is not None else "—"]
            for i, v in enumerate(vals):
                text, color = v if isinstance(v, tuple) else (v, TEXT)
                ctk.CTkLabel(line, text=text, width=self.COLS[i][1], anchor="w",
                             font=font(12, "bold" if i == 2 else "normal"),
                             text_color=color).grid(row=0, column=i, padx=(4 if i == 0 else 0, 0),
                                                    pady=4)
            ctk.CTkButton(line, text="Open", width=60, height=24, fg_color=CARD_HI,
                          hover_color=BORDER, font=font(11),
                          command=lambda i=r["index"]: app.show_frame(i)).grid(
                row=0, column=len(vals), padx=6)


class SettingsDialog(ctk.CTkToplevel):
    FIELDS = [("incoming", "Scans folder (incoming/)", str),
              ("adb", "adb executable", str),
              ("phone", "Phone ip:port (wireless debugging)", str),
              ("remote", "Open Camera folder on phone", str),
              ("n_photos", "Photos per scan (= Open Camera repeat)", int),
              ("settle", "Laser warm-up before shutter [s]", float),
              ("max_laser_s", "Max laser on-time [s] (< 10)", float),
              ("write_timeout", "File write timeout [s]", float)]

    def __init__(self, app: App):
        super().__init__(app, fg_color=PANEL)
        self.app = app
        self.title("Settings")
        self.geometry("520x560")
        self.transient(app)
        self.after(100, self.grab_set)
        ctk.CTkLabel(self, text="Capture settings", font=font(18, "bold")).pack(
            anchor="w", padx=20, pady=(18, 8))
        self.vars = {}
        cap = app.settings["capture"]
        for key, label, _ in self.FIELDS:
            val = app.settings["incoming"] if key == "incoming" else cap.get(key, "")
            ctk.CTkLabel(self, text=label, font=font(12), text_color=MUTED).pack(
                anchor="w", padx=20, pady=(6, 0))
            v = ctk.StringVar(value=str(val))
            ctk.CTkEntry(self, textvariable=v, fg_color=CARD, border_color=BORDER).pack(
                fill="x", padx=20)
            self.vars[key] = v
        self.del_var = ctk.BooleanVar(value=cap.get("delete_after_pull", True))
        ctk.CTkSwitch(self, text="Delete photos from phone after a verified pull",
                      variable=self.del_var, progress_color=ACCENT).pack(
            anchor="w", padx=20, pady=12)
        ctk.CTkButton(self, text="Save", fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      command=self.save).pack(fill="x", padx=20, pady=(0, 18))

    def save(self):
        cap = self.app.settings["capture"]
        try:
            for key, label, typ in self.FIELDS:
                raw = self.vars[key].get().strip()
                if key == "incoming":
                    self.app.settings["incoming"] = raw
                else:
                    cap[key] = typ(raw)
        except ValueError:
            self.app.set_status(f"Settings: '{label}' must be a number", "bad")
            return
        if cap["max_laser_s"] >= 10:
            cap["max_laser_s"] = 9.0
        cap["delete_after_pull"] = self.del_var.get()
        save_settings(self.app.settings)
        self.app._sync_capture_cfg()
        self.app.known = set(self.app._scan_dirs())
        self.app.refresh_scans()
        self.app.set_status("Settings saved", "ok")
        self.destroy()


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
