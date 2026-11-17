"""
AstroSense GUI - Machine Learning for Autonomous Spacecraft Diagnostics (Tkinter)

Pipeline (as in the synopsis / Phase-I report):
  clean broken rows -> Min-Max Scaling -> Isolation Forest learns the healthy pattern
  -> System Health Score (0-100%) -> automatic warning checklist.

Run:      python main.py
Install:  pip install pandas scikit-learn numpy
"""
import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

from .backend import *  # noqa: F401,F403  (FIELDS, FBY, Detector, clean, simulate, ...)

# ---------------------------------------------------------------- palette / fonts
BG, PANEL, PANEL2, PANEL3, LINE = "#0D1420", "#152131", "#1B2A3D", "#24364B", "#30445B"
INK, INK_DIM = "#EDF3FA", "#A4B5C8"
ACCENT, ACCENT_HOVER = "#5688E8", "#6D9CF2"
OK, WARN, DANGER = "#43C6A8", "#E9AD55", "#E66C7A"
DANGER_BG, INSIGHT_BG = "#35212B", "#1A2A40"
FONT_TITLE = ("Segoe UI", 22, "bold")
FONT_SUB = ("Segoe UI", 10)
FONT_H = ("Segoe UI", 12, "bold")
FONT_B = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_SMALL = ("Segoe UI", 9)
FONT_CAP = ("Segoe UI", 8, "bold")
FONT_MONO = ("Consolas", 10)

GROUPS = [
    ("power", "Power System", ""), ("propulsion", "Propulsion", ""),
    ("avionics", "Avionics", ""), ("cabin", "Cabin & Life Support", ""),
    ("attitude", "Attitude & Motion", ""), ("environment", "Space Environment", ""),
]
TAB_LABELS = {"power": "Power", "propulsion": "Propulsion", "avionics": "Avionics",
              "cabin": "Cabin", "attitude": "Attitude", "environment": "Environment"}

# short name of each measuring unit (the bottom box shows the units of the tab you are on)
UNIT_NAMES = {
    "V": "Volts", "W": "Watts", "bar": "Bar", "g/s": "Grams per second", "C": "Degrees Celsius",
    "kPa": "Kilopascals", "mmHg": "Millimetres of mercury", "deg/s": "Degrees per second",
    "RPM": "Revolutions per minute", "uSv/h": "Microsieverts per hour",
}


def units_text(group=None):
    """'V - Volts     W - Watts' for one tab's group, or every unit when group is None (Data tab)."""
    units = dict.fromkeys(f["unit"] for f in FIELDS if group in (None, f["group"]))
    return "   \u00B7   ".join(f"{u} - {UNIT_NAMES.get(u, u)}" for u in units)


# ---------------------------------------------------------------- widgets
def caption(parent, text, bg=BG, **pack):
    """Small upper-case section caption."""
    tk.Label(parent, text=text.upper(), bg=bg, fg=INK_DIM, font=FONT_CAP).pack(anchor="w", **pack)


class Tooltip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        if self.tip or not self.text:
            return
        x, y = self.widget.winfo_rootx() + 12, self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, bg="#1e2440", fg=INK, font=FONT_SMALL, padx=8, pady=4).pack()

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class HoverButton(tk.Button):
    def __init__(self, master, bg, hover_bg, fg="white", **kw):
        kw.setdefault("font", FONT_B)
        super().__init__(master, bg=bg, fg=fg, activebackground=hover_bg, activeforeground=fg,
                         relief="flat", bd=0, cursor="hand2", **kw)
        self.bind("<Enter>", lambda e: self.configure(bg=hover_bg))
        self.bind("<Leave>", lambda e: self.configure(bg=bg))


class RoundGauge(tk.Canvas):
    def __init__(self, master, size=140, **kw):
        kw.setdefault("bg", BG)
        super().__init__(master, width=size, height=size, highlightthickness=0, **kw)
        self.size, self.value = size, 0
        self._draw(0, OK)

    def _draw(self, value, color):
        self.delete("all")
        s, pad = self.size, 10
        self.create_oval(pad, pad, s - pad, s - pad, outline=PANEL3, width=10)
        if value > 0:
            self.create_arc(pad, pad, s - pad, s - pad, start=90, extent=-3.6 * value,
                            outline=color, width=10, style="arc")
        self.create_text(s / 2, s / 2 - 6, text=f"{value}%", fill=INK, font=("Segoe UI", 20, "bold"))
        self.create_text(s / 2, s / 2 + 18, text="health", fill=INK_DIM, font=FONT_SMALL)

    def animate_to(self, target, color, steps=18):
        start = self.value

        def tick(i):
            self.value = round(start + (target - start) * i / steps)
            self._draw(self.value, color)
            if i < steps:
                self.after(12, lambda: tick(i + 1))
        tick(0)


# ---------------------------------------------------------------- main app
class AstroSense(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AstroSense \u2014 Spacecraft Diagnostics Console")
        self.configure(bg=BG)
        self.geometry(f"1320x{min(900, self.winfo_screenheight() - 70)}")
        self.minsize(1050, 680)
        self.entries, self.scales, self._calc = {}, {}, None
        self.det, self.log_raw, self.log_name, self._tl = Detector(), None, "", None
        self._blink_job, self._blink_state = None, False
        self.trees_var, self.cont_var = tk.IntVar(value=200), tk.DoubleVar(value=0.01)
        self._build_style()
        self._build_menu()
        self._build_layout()
        self.use_builtin()
        self.load_sample()
        self.bind("<Return>", lambda e: self.run_diagnostics())
        self._tick_clock()

    # ---------- style / menu ----------
    def _build_style(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure("TNotebook", background=PANEL, borderwidth=0, tabmargins=(2, 2, 2, 0))
        st.configure("TNotebook.Tab", background=PANEL2, foreground=INK_DIM, padding=(16, 10),
                     font=FONT_BOLD, borderwidth=0)
        st.map("TNotebook.Tab", background=[("selected", ACCENT), ("active", PANEL3)],
               foreground=[("selected", "white"), ("active", INK)])
        st.configure("Dark.TSpinbox", fieldbackground=BG, background=PANEL3, foreground=INK,
                     arrowcolor=INK_DIM, bordercolor=LINE, lightcolor=LINE, darkcolor=LINE,
                     insertcolor=INK, padding=3)
        st.configure("Vertical.TScrollbar", background=PANEL3, troughcolor=BG, bordercolor=BG,
                     arrowcolor=INK_DIM, lightcolor=PANEL3, darkcolor=PANEL3)

    def _build_menu(self):
        bar = tk.Menu(self)
        m = tk.Menu(bar, tearoff=0)
        m.add_command(label="Load Healthy Training CSV...", command=self.load_baseline_csv)
        m.add_command(label="Use Built-in Baseline", command=self.use_builtin)
        m.add_command(label="Upload Telemetry Log CSV...", command=self.upload_log)
        m.add_command(label="Load OPS-SAT-AD dataset.csv (real data)...", command=self.load_opssat)
        m.add_separator()
        m.add_command(label="Save Sample Files...", command=self.save_samples)
        m.add_separator()
        m.add_command(label="Exit", command=self.destroy)
        bar.add_cascade(label="File", menu=m)
        h = tk.Menu(bar, tearoff=0)
        h.add_command(label="How the score works", command=self._show_method)
        h.add_command(label="Ranges, equations & sources", command=self._show_sources)
        bar.add_cascade(label="Help", menu=h)
        self.config(menu=bar)

    def _show_method(self):
        messagebox.showinfo(
            "How the score works",
            "1. Broken rows (missing / non-numeric values) are removed.\n"
            "2. Min-Max Scaling puts every sensor on a 0-1 scale.\n"
            "3. Isolation Forest learns the healthy pattern from the training data.\n"
            "4. Sensor health: every sensor is compared with its nominal range. Inside the range = 100 %; "
            "outside, it falls in a straight line to 0 % at the sensor's hard limit. The sensor healths are "
            "multiplied (a series system: every part must work).\n"
            "5. Pattern health: the Isolation Forest anomaly score is converted to 0-100 % "
            "(100 = as healthy as training data, 80 = flagging threshold, 0 = every sensor at its limit).\n"
            "   Record health = the lower of steps 4 and 5.\n"
            "   System Health Score = mean record health x share of unflagged records.\n"
            "6. For flagged records, the sensors that deviate most (in standard deviations "
            "from the healthy mean) become the warning checklist.\n\n"
            "Prototype: the built-in baseline is synthetic. Use real healthy telemetry for real results.")

    # ---------- layout ----------
    def _build_layout(self):
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=28, pady=(22, 16))
        logo = tk.Canvas(head, width=34, height=34, bg=BG, highlightthickness=0)
        logo.create_oval(3, 3, 31, 31, outline=ACCENT, width=2)
        logo.create_oval(12, 12, 22, 22, fill=ACCENT, outline="")
        logo.pack(side="left", padx=(0, 12))
        titles = tk.Frame(head, bg=BG)
        titles.pack(side="left")
        tk.Label(titles, text="AstroSense", bg=BG, fg=INK, font=FONT_TITLE).pack(anchor="w")
        tk.Label(titles, text="Isolation Forest diagnostics for spacecraft telemetry", bg=BG, fg=INK_DIM,
                 font=FONT_SUB).pack(anchor="w")
        self.clock_lbl = tk.Label(head, bg=BG, fg=INK_DIM, font=FONT_MONO)
        self.clock_lbl.pack(side="right")
        dot = tk.Canvas(head, width=10, height=10, bg=BG, highlightthickness=0)
        dot.create_oval(1, 1, 9, 9, fill=OK, outline="")
        dot.pack(side="right", padx=8)
        tk.Frame(self, bg=LINE, height=1).pack(fill="x", padx=28)

        self._build_units_footer()
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=28, pady=(18, 14))
        body.columnconfigure(0, weight=6)
        body.columnconfigure(1, weight=5)
        body.rowconfigure(0, weight=1)
        self._build_input_panel(body)
        self._build_result_panel(body)

    def _build_units_footer(self):
        """Box at the bottom: measuring units used on the tab that is open."""
        foot = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        foot.pack(side="bottom", fill="x", padx=28, pady=(0, 18))
        tk.Label(foot, text="UNITS", bg=PANEL, fg=ACCENT, font=FONT_CAP).pack(side="left", anchor="n",
                                                                           padx=(14, 12), pady=(10, 0))
        self.units_lbl = tk.Label(foot, bg=PANEL, fg=INK, font=FONT_SMALL, anchor="w", justify="left",
                                  wraplength=900)
        self.units_lbl.pack(side="left", fill="x", expand=True, padx=(0, 14), pady=8)
        foot.bind("<Configure>", lambda e: self.units_lbl.configure(wraplength=max(200, e.width - 110)))

    def _update_units(self, _=None):
        i = self.nb.index(self.nb.select())
        self.units_lbl.configure(text="Open a sensor tab (Power, Propulsion, ...) to see its measuring units."
                                 if i == 0 else units_text(GROUPS[i - 1][0]))

    def _build_input_panel(self, parent):
        left = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        tk.Label(left, text="Telemetry Input", bg=PANEL, fg=INK, font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=20, pady=(18, 4))
        sub = tk.Label(left, text="Data tab: train on healthy data and check an uploaded log.   "
                                  "Sensor tabs: check a single manual reading.",
                       bg=PANEL, fg=INK_DIM, font=FONT_SMALL, justify="left", wraplength=560)
        sub.pack(anchor="w", padx=20, pady=(0, 12))
        left.bind("<Configure>", lambda e: sub.configure(wraplength=max(200, e.width - 40)))
        tk.Label(left, text="Press Enter to run diagnostics for the open tab.", bg=PANEL, fg=INK_DIM,
                 font=FONT_SMALL).pack(side="bottom", anchor="w", padx=20, pady=(0, 14))
        btns = tk.Frame(left, bg=PANEL)
        btns.pack(side="bottom", fill="x", padx=20, pady=(12, 10))
        tk.Frame(left, bg=LINE, height=1).pack(side="bottom", fill="x")
        HoverButton(btns, ACCENT, ACCENT_HOVER, text="Run Diagnostics", font=FONT_BOLD,
                    padx=18, pady=9, command=self.run_diagnostics).pack(side="left", padx=(0, 10))
        for txt, cmd in (("Nominal Sample", self.load_sample), ("Simulate Fault", self.load_fault),
                         ("Clear", self.clear_all)):
            HoverButton(btns, PANEL3, LINE, fg=INK, text=txt, padx=14, pady=9, command=cmd).pack(side="left", padx=(0, 8))
        self.nb = ttk.Notebook(left)
        self.nb.pack(fill="both", expand=True, padx=16, pady=(4, 10))
        self.nb.bind("<<NotebookTabChanged>>", self._update_units)
        data = tk.Frame(self.nb, bg=PANEL)
        self.nb.add(data, text="Data")
        self._build_data_tab(data)
        for gid, gname, icon in GROUPS:
            tab = tk.Frame(self.nb, bg=PANEL)
            self.nb.add(tab, text=TAB_LABELS.get(gid, gname.split(" ")[0]))
            for f in [f for f in FIELDS if f["group"] == gid]:
                self._build_field_row(tab, f)

        self._update_units()

    def _build_data_tab(self, tab):
        def section(n, t):
            h = tk.Frame(tab, bg=PANEL)
            h.pack(fill="x", padx=16, pady=(16, 4))
            tk.Label(h, text=str(n), bg=ACCENT, fg="white", font=FONT_CAP, width=2).pack(side="left", padx=(0, 8))
            tk.Label(h, text=t, bg=PANEL, fg=INK, font=FONT_H).pack(side="left")

        def row():
            r = tk.Frame(tab, bg=PANEL)
            r.pack(fill="x", padx=16, pady=5)
            return r

        section(1, "Healthy baseline (model training)")
        self.base_lbl = tk.Label(tab, bg=PANEL, fg=INK_DIM, font=FONT_SMALL, wraplength=520, justify="left")
        self.base_lbl.pack(anchor="w", padx=16)
        r = row()
        for txt, cmd in (("Load healthy CSV...", self.load_baseline_csv), ("Use built-in baseline", self.use_builtin),
                         ("Retrain", self.retrain)):
            HoverButton(r, PANEL3, LINE, fg=INK, text=txt, padx=12, pady=6, command=cmd).pack(side="left", padx=(0, 6))
        r = row()
        tk.Label(r, text="Trees", bg=PANEL, fg=INK_DIM, font=FONT_SMALL).pack(side="left")
        ttk.Spinbox(r, from_=50, to=500, increment=50, textvariable=self.trees_var, width=6,
                    style="Dark.TSpinbox").pack(side="left", padx=(6, 18))
        tk.Label(r, text="Expected anomalies", bg=PANEL, fg=INK_DIM, font=FONT_SMALL).pack(side="left")
        ttk.Spinbox(r, from_=0.001, to=0.2, increment=0.005, format="%.3f", textvariable=self.cont_var, width=7,
                    style="Dark.TSpinbox").pack(side="left", padx=6)

        section(2, "Telemetry log to check")
        r = row()
        HoverButton(r, ACCENT, ACCENT_HOVER, text="Upload log CSV...", font=FONT_BOLD,
                    padx=14, pady=7, command=self.upload_log).pack(side="left")
        self.log_lbl = tk.Label(tab, text="No log uploaded yet.", bg=PANEL, fg=INK_DIM, font=FONT_SMALL,
                                wraplength=520, justify="left")
        self.log_lbl.pack(anchor="w", padx=16)

        section(3, "Health score over time")
        self.tl = tk.Canvas(tab, bg=BG, height=130, highlightthickness=1, highlightbackground=LINE)
        self.tl.pack(fill="x", padx=16, pady=(2, 10))
        self.tl.bind("<Configure>", lambda e: self._draw_timeline())

        def rewrap(e):
            for w in (self.base_lbl, self.log_lbl):
                w.configure(wraplength=max(200, e.width - 36))
        tab.bind("<Configure>", rewrap)

    def _build_field_row(self, parent, f):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=16, pady=(12, 0))
        top = tk.Frame(row, bg=PANEL)
        top.pack(fill="x")
        lbl = tk.Label(top, text=f["label"], bg=PANEL, fg=INK, font=FONT_BOLD, cursor="question_arrow")
        lbl.pack(side="left")
        Tooltip(lbl, f'Range: {f["min"]}\u2013{f["max"]} {f["unit"]}\n{f["src"][:110]}')
        var = tk.StringVar()
        tk.Entry(top, textvariable=var, width=10, bg=BG, fg=INK, insertbackground=INK, relief="flat",
                 highlightthickness=1, highlightbackground=LINE, highlightcolor=ACCENT_HOVER, font=FONT_MONO,
                 justify="right").pack(side="right", ipady=2)
        tk.Label(top, text=f["unit"], bg=PANEL, fg=INK_DIM, font=FONT_SMALL).pack(side="right", padx=(4, 8))
        tk.Label(row, text=f'Nominal {f["min"]}\u2013{f["max"]} {f["unit"]}', bg=PANEL, fg=INK_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        lo, hi = f["hard_min"], f["hard_max"]
        scale = tk.Scale(row, from_=lo, to=hi, orient="horizontal", resolution=(hi - lo) / 200, showvalue=False,
                         bg=PANEL, troughcolor="#354A63", highlightthickness=0, bd=0, sliderrelief="flat",
                         length=300, activebackground=ACCENT)
        scale.pack(fill="x", pady=(4, 0))
        tk.Frame(parent, bg=LINE, height=1).pack(fill="x", padx=16, pady=(10, 0))
        busy = {"f": False}

        def sync_from_scale():              # only when the USER moves the slider: copy its value into the box
            busy["f"] = True
            var.set(f"{scale.get():.4g}")
            busy["f"] = False

        def on_entry(*_):                   # typing: the slider just follows; the typed text is never changed
            if not busy["f"]:
                try:
                    scale.set(float(var.get()))
                except ValueError:
                    pass

        for ev in ("<Button-1>", "<B1-Motion>", "<ButtonRelease-1>", "<KeyRelease>"):
            scale.bind(ev, lambda e: scale.after(1, sync_from_scale), add="+")
        var.trace_add("write", on_entry)
        self.entries[f["key"]], self.scales[f["key"]] = var, scale

    def _build_result_panel(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        top = tk.Frame(right, bg=BG)
        top.pack(fill="x", pady=(0, 10))
        self.status_card = tk.Frame(top, bg=PANEL2, highlightthickness=1, highlightbackground=LINE)
        self.status_card.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.status_dot = tk.Canvas(self.status_card, width=14, height=14, bg=PANEL2, highlightthickness=0)
        self.status_dot.pack(anchor="w", padx=20, pady=(20, 0))
        self.status_title = tk.Label(self.status_card, text="Awaiting data", bg=PANEL2, fg=INK,
                                     font=("Segoe UI", 14, "bold"))
        self.status_title.pack(anchor="w", padx=20, pady=(8, 0))
        self.status_sub = tk.Label(self.status_card, text="Upload a log or enter a reading, then run diagnostics.",
                                   bg=PANEL2, fg=INK_DIM, font=FONT_B, wraplength=300, justify="left")
        self.status_sub.pack(anchor="w", padx=20, pady=(4, 20))
        gf = tk.Frame(top, bg=PANEL2, highlightthickness=1, highlightbackground=LINE)
        gf.pack(side="left")
        self.gauge = RoundGauge(gf, size=150, bg=PANEL2)
        self.gauge.pack(padx=18, pady=(16, 8))
        HoverButton(gf, PANEL3, LINE, fg=INK, text="Show calculation", font=FONT_SMALL, padx=10, pady=4,
                    command=self._show_calc).pack(pady=(0, 14))

        stats = tk.Frame(right, bg=BG)
        stats.pack(fill="x", pady=(0, 6))
        self.stat_health = self._stat_box(stats, "System status", "\u2014")
        self.stat_flags = self._stat_box(stats, "Flagged records", "0")
        self.stat_quality = self._stat_box(stats, "Input quality", "\u2014", last=True)

        caption(right, "Warning checklist", pady=(12, 6))
        self.chk_box = tk.Frame(right, bg=BG)
        self.chk_box.pack(fill="x", pady=(0, 4))
        tk.Label(self.chk_box, text="No analysis yet.", bg=BG, fg=INK_DIM, font=FONT_B).pack(anchor="w")

        caption(right, "Subsystem status", pady=(12, 6))
        sub_card = tk.Frame(right, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        sub_card.pack(fill="x")
        self.sub_box = tk.Frame(sub_card, bg=PANEL)
        self.sub_box.pack(fill="x", padx=14, pady=8)
        self.sub_box.columnconfigure((0, 1), weight=1)

        self.insight_frame = tk.Frame(right, bg=INSIGHT_BG, highlightthickness=1, highlightbackground=ACCENT)
        self.insight_frame.pack(fill="x", pady=(14, 0))
        self.insight = tk.Label(self.insight_frame, text="No readings screened yet.", bg=INSIGHT_BG,
                                fg=INK_DIM, font=FONT_B, wraplength=470, justify="left")
        self.insight.pack(anchor="w", padx=14, pady=12)
        self._set_dot(OK)

        def rewrap(e):
            self.insight.configure(wraplength=max(200, e.width - 48))
            self.status_sub.configure(wraplength=max(160, e.width - 240))
        right.bind("<Configure>", rewrap)

    def _stat_box(self, parent, title, value, last=False):
        box = tk.Frame(parent, bg=PANEL2, highlightthickness=1, highlightbackground=LINE)
        box.pack(side="left", fill="both", expand=True, padx=(0, 0 if last else 10))
        tk.Label(box, text=title.upper(), bg=PANEL2, fg=INK_DIM, font=FONT_CAP).pack(anchor="w", padx=14, pady=(12, 0))
        v = tk.Label(box, text=value, bg=PANEL2, fg=INK, font=("Segoe UI", 14, "bold"))
        v.pack(anchor="w", padx=14, pady=(4, 12))
        return v

    def _set_dot(self, color):
        self.status_dot.delete("all")
        self.status_dot.create_oval(1, 1, 13, 13, fill=color, outline="")

    def _tick_clock(self):
        self.clock_lbl.configure(text=time.strftime("%b %d, %Y   %H:%M:%S"))
        self.after(1000, self._tick_clock)

    def _start_blink(self):
        self._stop_blink()

        def blink():
            self._blink_state = not self._blink_state
            self._set_dot(DANGER if self._blink_state else "#5c1f2c")
            self._blink_job = self.after(500, blink)
        blink()

    def _stop_blink(self):
        if self._blink_job:
            self.after_cancel(self._blink_job)
            self._blink_job = None

    # ---------- manual entry helpers ----------
    def load_sample(self):
        for f in FIELDS:
            self.entries[f["key"]].set(f'{f["default"]}')
        if self.nb.index(self.nb.select()) == 0:
            self.nb.select(1)

    def load_fault(self):
        self.load_sample()
        for k, v in FAULT_OVERRIDES.items():
            self.entries[k].set(str(v))
        self.nb.select(1)

    def clear_all(self):
        for f in FIELDS:
            self.entries[f["key"]].set("")

    def _parse(self, key):
        try:
            return float(self.entries[key].get().strip())
        except (ValueError, KeyError):
            return None

    # ---------- training ----------
    def train(self, df, label):
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            self.det.fit(df, int(self.trees_var.get()), float(self.cont_var.get()))
        except Exception as e:
            messagebox.showerror("Training failed", str(e))
            return False
        finally:
            self.config(cursor="")
        self.base_df, self.base_name = df, label
        self.base_lbl.configure(text=f"{label}\n{len(df)} rows \u00B7 {len(df.columns)} sensors \u00B7 "
                                     f"Isolation Forest ({self.trees_var.get()} trees), Min-Max scaled")
        return True

    def use_builtin(self):
        self.train(make_baseline(), "Physics-based simulated healthy baseline (Help > Ranges, equations & sources)")

    def retrain(self):
        self.train(self.base_df, self.base_name)

    def load_baseline_csv(self):
        path = filedialog.askopenfilename(title="Healthy telemetry CSV", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            df = read_csv(path)
            tc = time_col(df)
            cols = [c for c in df.columns if c != tc and pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.8]
            df, dropped = clean(df, cols)
            if not cols or len(df) < 50:
                raise ValueError("Need at least 50 clean rows and one numeric sensor column.")
            self.train(df[cols].reset_index(drop=True), f"{os.path.basename(path)} ({dropped} broken rows removed)")
        except Exception as e:
            messagebox.showerror("Could not load training data", str(e))

    # ---------- log upload / diagnostics ----------
    def upload_log(self):
        path = filedialog.askopenfilename(title="Telemetry log CSV", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            self.log_raw, self.log_name = read_csv(path), os.path.basename(path)
        except Exception as e:
            messagebox.showerror("Could not read file", str(e))
            return
        self.nb.select(0)
        self._run_log()

    def run_diagnostics(self):
        self._run_log() if self.nb.index(self.nb.select()) == 0 else self._run_manual()

    def _run_log(self):
        if self.log_raw is None:
            messagebox.showinfo("No log", "Upload a telemetry log CSV first (Data tab).")
            return
        missing = [c for c in self.det.cols if c not in self.log_raw.columns]
        if missing:
            messagebox.showerror("Missing columns", "The log has no column for: " + ", ".join(missing))
            return
        df, dropped = clean(self.log_raw, self.det.cols)
        if df.empty:
            messagebox.showerror("Empty log", "No valid rows left after cleaning.")
            return
        df = df.reset_index(drop=True)
        tc = time_col(df)
        t0, t1 = (str(df[tc].iloc[0]), str(df[tc].iloc[-1])) if tc else ("row 0", f"row {len(df) - 1}")
        self.log_lbl.configure(text=f"{self.log_name} \u00B7 {len(self.log_raw)} rows \u00B7 {dropped} broken rows removed "
                                    f"\u00B7 {len(df)} analysed")
        labels = df[tc].astype(str).tolist() if tc else [f"row {i}" for i in range(len(df))]
        self._show(df, f"{dropped} rows dropped", t0, t1, labels, self.log_name)

    def _run_manual(self):
        row, filled = {}, 0
        for c in self.det.cols:
            v = self._parse(c)
            filled += v is not None
            row[c] = v if v is not None else float(self.det.mean[c])   # blank field -> assumed nominal
        if filled == 0:
            messagebox.showinfo("No readings", "Enter at least one reading first.")
            return
        self._show(pd.DataFrame([row]), f"{filled}/{len(self.det.cols)} fields", None, None, ["this reading"])

    def _show(self, df, quality, t0, t1, labels, source="this reading"):
        health, flagged, z = self.det.score(df)
        n, frac = len(df), float(flagged.mean())
        overall = int(round(health.mean() if n < 30 else health.mean() * (1 - frac)))
        self._calc = self.det.report(df, flagged, overall, source)
        is_anom = bool(flagged.any()) if n < 30 else frac > 0.03
        items = self.det.explain(flagged, z) if flagged.any() else []
        color = OK if overall >= 80 else (WARN if overall >= 55 else DANGER)

        self.gauge.animate_to(overall, color)
        bg = DANGER_BG if is_anom else PANEL2
        self.status_card.configure(bg=bg, highlightbackground=DANGER if is_anom else OK)
        for w in (self.status_dot, self.status_title, self.status_sub):
            w.configure(bg=bg)
        self.status_title.configure(text="Anomaly detected" if is_anom else "No anomaly detected", fg=DANGER if is_anom else OK)
        self.status_sub.configure(text=f"{int(flagged.sum())} of {n} record(s) deviate from the learned healthy pattern."
                                  if is_anom else "Data matches the learned healthy pattern.")
        self._stop_blink()
        self._start_blink() if is_anom else self._set_dot(OK)
        self.stat_health.configure(text="ANOMALOUS" if is_anom else "NOMINAL", fg=DANGER if is_anom else OK)
        self.stat_flags.configure(text=f"{int(flagged.sum())} / {n}")
        self.stat_quality.configure(text=quality)

        if t0 is not None:
            self._tl = (np.asarray(health), np.asarray(flagged), t0, t1)
            self._draw_timeline()

        # warning checklist
        for w in self.chk_box.winfo_children():
            w.destroy()
        if not items:
            tk.Label(self.chk_box, text="No warnings \u2014 all systems nominal.", bg=BG, fg=OK, font=FONT_B).pack(anchor="w")
        for col, cnt, mz in items[:6]:
            c = DANGER if abs(mz) >= 4 else WARN
            r = tk.Frame(self.chk_box, bg=DANGER_BG)
            r.pack(fill="x", pady=3)
            tk.Frame(r, bg=c, width=3).pack(side="left", fill="y")
            lab = tk.Label(r, bg=DANGER_BG, fg=INK, font=FONT_SMALL, anchor="w", padx=8, pady=5, cursor="hand2",
                           text=f"\u2610  {warning_text(col, mz)}   \u00B7  {cnt} record(s), {mz:+.1f}\u03C3")
            lab.pack(side="left", fill="x")
            lab.bind("<Button-1>", lambda e, l=lab: l.configure(
                text=("\u2611" if l.cget("text")[0] == "\u2610" else "\u2610") + l.cget("text")[1:]))
        if len(items) > 6:
            tk.Label(self.chk_box, text=f"\u2026and {len(items) - 6} more", bg=BG, fg=INK_DIM, font=FONT_SMALL).pack(anchor="w")

        # subsystem status
        gstate = {g: 0 for g, _, _ in GROUPS}
        for col, cnt, mz in items:
            if col in FBY:
                g = FBY[col]["group"]
                gstate[g] = max(gstate[g], 2 if abs(mz) >= 4 else 1)
        for w in self.sub_box.winfo_children():
            w.destroy()
        for i, (gid, gname, icon) in enumerate(GROUPS):
            s = gstate[gid]
            c, lab = (OK, "Normal") if s == 0 else ((WARN, "Watch") if s == 1 else (DANGER, "Fault"))
            cell = tk.Frame(self.sub_box, bg=PANEL)
            cell.grid(row=i // 2, column=i % 2, sticky="ew", pady=4, padx=(0, 12))
            d = tk.Canvas(cell, width=10, height=10, bg=PANEL, highlightthickness=0)
            d.create_oval(2, 2, 10, 10, fill=c, width=0)
            d.pack(side="left", padx=(0, 6))
            tk.Label(cell, text=gname, bg=PANEL, fg=INK, font=FONT_SMALL).pack(side="left")
            tk.Label(cell, text=lab, bg=PANEL, fg=c, font=("Segoe UI", 9, "bold")).pack(side="right")

        if is_anom and items:
            first = labels[int(np.flatnonzero(flagged)[0])]
            text = (f"{warning_text(items[0][0], items[0][2])} is the strongest signal. First flagged at "
                    f"{first}. Inspect this subsystem before proceeding.")
        elif is_anom:
            text = "Unusual combination of readings detected; no single sensor stands out."
        else:
            text = f"{n} record(s) screened against the healthy baseline; nothing unusual found."
        self.insight_frame.configure(highlightbackground=DANGER if is_anom else ACCENT)
        self.insight.configure(text=text, fg=INK)

    def _draw_timeline(self):
        c = self.tl
        c.delete("all")
        w, h = max(c.winfo_width(), 300), int(c["height"])
        pl, pr, pt, pb = 34, 10, 10, 24
        y_of = lambda v: pt + (h - pt - pb) * (1 - v / 100)
        for v in (0, 50, 100):
            c.create_line(pl, y_of(v), w - pr, y_of(v), fill=LINE)
            c.create_text(pl - 6, y_of(v), text=str(v), fill=INK_DIM, font=("Segoe UI", 8), anchor="e")
        if not self._tl:
            c.create_text(w / 2, h / 2, text="Upload a log to see its health curve", fill=INK_DIM, font=FONT_SMALL)
            return
        hl, fl, t0, t1 = self._tl
        n = len(hl)
        c.create_line(pl, y_of(80), w - pr, y_of(80), fill=WARN, dash=(4, 3))
        idx = [b[int(np.argmin(hl[b]))] for b in np.array_split(np.arange(n), min(n, w - pl - pr))]
        x_of = lambda i: pl + (w - pl - pr) * (i / max(n - 1, 1))
        if len(idx) > 1:
            c.create_line(*[p for i in idx for p in (x_of(i), y_of(hl[i]))], fill=ACCENT, width=1.5)
        for i in idx:
            if fl[i]:
                c.create_oval(x_of(i) - 2.5, y_of(hl[i]) - 2.5, x_of(i) + 2.5, y_of(hl[i]) + 2.5, fill=DANGER, outline="")
        c.create_text(pl, h - 8, text=t0, fill=INK_DIM, font=("Segoe UI", 8), anchor="w")
        c.create_text(w - pr, h - 8, text=t1, fill=INK_DIM, font=("Segoe UI", 8), anchor="e")

    def _show_calc(self):
        if not self._calc:
            messagebox.showinfo("Nothing calculated yet", "Run diagnostics first, then open the calculation.")
            return
        win = tk.Toplevel(self, bg=PANEL)
        win.title("How the health % was calculated")
        win.geometry("1000x680")
        txt = tk.Text(win, bg=BG, fg=INK, font=FONT_MONO, wrap="none", relief="flat", padx=14, pady=10)
        sb = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("end", self._calc)
        txt.configure(state="disabled")

    def _show_sources(self):
        win = tk.Toplevel(self, bg=PANEL)
        win.title("Ranges, equations & sources")
        win.geometry("920x640")
        txt = tk.Text(win, bg=BG, fg=INK, font=FONT_SMALL, wrap="word", relief="flat", padx=14, pady=10)
        txt.pack(fill="both", expand=True)
        txt.insert("end", EQUATIONS + "\n\nSENSOR RANGES AND WHERE THEY COME FROM\n")
        for f in FIELDS:
            txt.insert("end", f'\n{f["label"]} [{f["unit"]}]: {f["min"]} to {f["max"]}\n    {f["src"]}\n')
        txt.configure(state="disabled")

    def load_opssat(self):
        """Real telemetry: OPSSAT-AD dataset.csv (Zenodo 12588359). Trains on normal segments, tests on the rest."""
        path = filedialog.askopenfilename(title="OPSSAT-AD dataset.csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            from sklearn.metrics import precision_recall_fscore_support as prf
            df = pd.read_csv(path)
            low = {c.lower(): c for c in df.columns}
            lab = low.get("anomaly") or low.get("label")
            if lab is None:
                raise ValueError("No 'anomaly' label column found. Columns: " + ", ".join(map(str, df.columns)))
            skip = {lab.lower(), "train", "segment", "channel", "sampling", "id", "index"}
            feats = [c for c in df.columns if c.lower() not in skip and pd.api.types.is_numeric_dtype(df[c])]
            y = df[lab].astype(int)
            is_tr = (df[low["train"]].astype(int) == 1) if "train" in low else pd.Series(np.arange(len(df)) % 10 < 7, index=df.index)
            tr, dropped = clean(df[is_tr & (y == 0)], feats)
            te, _ = clean(df[~(is_tr & (y == 0))], feats)
            if len(tr) < 50 or te.empty:
                raise ValueError("Not enough clean training / test rows.")
            if not self.train(tr[feats].reset_index(drop=True), f"OPSSAT-AD real telemetry ({os.path.basename(path)}) \u2013 normal segments only"):
                return
            _, flagged, _ = self.det.score(te)
            truth = (te[lab].astype(int) != 0).to_numpy()
            p, r, f1, _ = prf(truth, flagged, average="binary", zero_division=0)
            self.log_raw, self.log_name = te.reset_index(drop=True), "OPSSAT-AD held-out segments"
            messagebox.showinfo("OPSSAT-AD evaluation",
                                f"Trained on {len(tr)} normal segments ({len(feats)} features).\n"
                                f"Tested on {len(te)} held-out segments ({int(truth.sum())} anomalous).\n\n"
                                f"Precision {p:.2f}   Recall {r:.2f}   F1 {f1:.2f}")
            self.nb.select(0)
            self._run_log()
        except Exception as e:
            messagebox.showerror("Could not load OPSSAT-AD", str(e))

    def save_samples(self):
        d = filedialog.askdirectory(title="Folder for sample CSV files")
        if not d:
            return
        make_baseline(3000, 99).to_csv(os.path.join(d, "sample_healthy_training.csv"), index=False)
        make_sample_log().to_csv(os.path.join(d, "sample_telemetry_log.csv"), index=False)
        messagebox.showinfo("Saved", "Saved sample_healthy_training.csv and sample_telemetry_log.csv.\n"
                                     "Load the healthy file as the baseline, then upload the log.")

