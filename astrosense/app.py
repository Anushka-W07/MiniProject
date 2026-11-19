"""
AstroSense GUI - Machine Learning for Autonomous Spacecraft Diagnostics (Tkinter)

Pipeline (as in the synopsis / Phase-I report):
  clean broken rows -> Min-Max Scaling -> Isolation Forest learns the healthy pattern
  -> System Health Score (0-100%) -> automatic warning checklist.

Pages:  Dashboard (overview) | Data (train + upload log) | Diagnostics (manual sensor tabs + report) | About
Run:      python main.py
Install:  pip install pandas scikit-learn numpy
"""
import os
import time
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont

import numpy as np
import pandas as pd

from .backend import *  # noqa: F401,F403  (FIELDS, FBY, Detector, clean, simulate, ...)

# ---------------------------------------------------------------- palette / fonts
BG, SIDEBAR = "#060b1a", "#081024"
CARD, CARD2, LINE, TRACK = "#0d1730", "#111f3f", "#1d2d57", "#1a2a52"
PANEL, PANEL2, PANEL3 = CARD, CARD2, "#1b2a52"
INK, INK_DIM = "#f1f5ff", "#9aa9c9"
ACCENT, ACCENT_HOVER, ACTIVE_BG = "#3d8bff", "#62a5ff", "#14357a"
OK, WARN, DANGER, PURPLE = "#2fd47a", "#f59a3c", "#ff5c7a", "#8272ff"
DANGER_BG, INSIGHT_BG = "#2a1522", "#0f1c3d"
FONT_WELCOME = ("Segoe UI", 24, "bold")
FONT_TITLE = ("Segoe UI", 26, "bold")
FONT_ITALIC = ("Segoe UI", 12, "italic")
FONT_SUB = ("Segoe UI", 10)
FONT_H = ("Segoe UI", 13, "bold")
FONT_B = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_SMALL = ("Segoe UI", 9)
FONT_SMALL_B = ("Segoe UI", 9, "bold")
FONT_CAP = ("Segoe UI", 8, "bold")
FONT_MONO = ("Consolas", 10)
OV_H = 230          # height of the dashboard health chart

GROUPS = [
    ("power", "Power System", ""), ("propulsion", "Propulsion", ""),
    ("avionics", "Avionics", ""), ("cabin", "Cabin & Life Support", ""),
    ("attitude", "Attitude & Motion", ""), ("environment", "Space Environment", ""),
]
TAB_LABELS = {"power": "Power", "propulsion": "Propulsion", "avionics": "Avionics",
              "cabin": "Cabin", "attitude": "Attitude", "environment": "Environment"}
GROUP_ICON = {"power": "bolt", "propulsion": "rocket", "avionics": "chip",
              "cabin": "life", "attitude": "target", "environment": "globe"}

METHOD_TEXT = (
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




# ---------------------------------------------------------------- drawing helpers
def blend(fg, bg, a):
    """Colour fg at opacity a over colour bg (Tk has no alpha)."""
    f = [int(fg[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x * a + y * (1 - a)) for x, y in zip(f, b))


def round_pts(x1, y1, x2, y2, r):
    """Polygon points of a rounded rectangle (draw with smooth=True)."""
    return [x1 + r, y1, x1 + r, y1, x2 - r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y1 + r, x2, y2 - r,
            x2, y2 - r, x2, y2, x2 - r, y2, x2 - r, y2, x1 + r, y2, x1 + r, y2, x1, y2, x1, y2 - r,
            x1, y2 - r, x1, y1 + r, x1, y1 + r, x1, y1]


def draw_icon(c, kind, cx, cy, s, color, w=2):
    """Small line/solid icons drawn with canvas shapes (no image files, no emoji)."""
    kw = dict(fill=color, outline=color)
    ln = dict(fill=color, width=w, capstyle="round", joinstyle="round")
    if kind == "home":
        c.create_polygon(cx - s, cy, cx, cy - s, cx + s, cy, **kw)
        c.create_rectangle(cx - s * .65, cy, cx + s * .65, cy + s * .9, **kw)
    elif kind == "data":
        c.create_oval(cx - s * .85, cy - s * .95, cx + s * .85, cy - s * .35, **kw)
        c.create_rectangle(cx - s * .85, cy - s * .65, cx + s * .85, cy + s * .65, **kw)
        c.create_oval(cx - s * .85, cy + s * .35, cx + s * .85, cy + s * .95, **kw)
    elif kind == "tools":
        c.create_line(cx - s, cy - s, cx + s, cy + s, **dict(ln, width=w + 1))
        c.create_line(cx - s, cy + s, cx + s, cy - s, **dict(ln, width=w + 1))
    elif kind == "info":
        c.create_oval(cx - s, cy - s, cx + s, cy + s, outline=color, width=w)
        c.create_text(cx, cy, text="i", fill=color, font=("Segoe UI", max(8, int(s * 1.1)), "bold"))
    elif kind == "bolt":
        c.create_polygon(cx + s * .2, cy - s, cx - s * .6, cy + s * .15, cx - s * .05, cy + s * .15,
                         cx - s * .25, cy + s, cx + s * .65, cy - s * .2, cx + s * .05, cy - s * .2, **kw)
    elif kind == "rocket":
        c.create_polygon(cx, cy - s, cx + s * .4, cy - s * .2, cx + s * .4, cy + s * .5,
                         cx - s * .4, cy + s * .5, cx - s * .4, cy - s * .2, **kw)
        c.create_polygon(cx - s * .4, cy + s * .1, cx - s * .85, cy + s * .7, cx - s * .4, cy + s * .5, **kw)
        c.create_polygon(cx + s * .4, cy + s * .1, cx + s * .85, cy + s * .7, cx + s * .4, cy + s * .5, **kw)
        c.create_line(cx, cy + s * .55, cx, cy + s, **ln)
    elif kind == "chip":
        c.create_rectangle(cx - s * .6, cy - s * .6, cx + s * .6, cy + s * .6, outline=color, width=w)
        for k in (-.3, .3):
            c.create_line(cx + k * s, cy - s * .6, cx + k * s, cy - s, **ln)
            c.create_line(cx + k * s, cy + s * .6, cx + k * s, cy + s, **ln)
            c.create_line(cx - s * .6, cy + k * s, cx - s, cy + k * s, **ln)
            c.create_line(cx + s * .6, cy + k * s, cx + s, cy + k * s, **ln)
    elif kind == "globe":
        c.create_oval(cx - s, cy - s, cx + s, cy + s, outline=color, width=w)
        c.create_oval(cx - s * .45, cy - s, cx + s * .45, cy + s, outline=color, width=w - 1 or 1)
        c.create_line(cx - s, cy, cx + s, cy, fill=color, width=w - 1 or 1)
    elif kind == "life":
        c.create_oval(cx - s, cy - s, cx + s, cy + s, outline=color, width=w)
        c.create_line(cx - s * .5, cy, cx + s * .5, cy, **ln)
        c.create_line(cx, cy - s * .5, cx, cy + s * .5, **ln)
    elif kind == "target":
        c.create_oval(cx - s, cy - s, cx + s, cy + s, outline=color, width=w)
        c.create_oval(cx - s * .4, cy - s * .4, cx + s * .4, cy + s * .4, outline=color, width=w)
        c.create_line(cx - s * 1.2, cy, cx - s * .6, cy, **ln)
        c.create_line(cx + s * .6, cy, cx + s * 1.2, cy, **ln)
    elif kind == "pulse":
        c.create_line(cx - s, cy, cx - s * .45, cy, cx - s * .2, cy - s * .8, cx + s * .15, cy + s * .8,
                      cx + s * .4, cy, cx + s, cy, **ln)
    elif kind == "shield":
        c.create_polygon(cx, cy - s, cx + s * .8, cy - s * .6, cx + s * .8, cy + s * .1, cx, cy + s,
                         cx - s * .8, cy + s * .1, cx - s * .8, cy - s * .6, **kw)
        c.create_line(cx - s * .4, cy, cx - s * .1, cy + s * .35, cx + s * .45, cy - s * .3,
                      fill=CARD, width=w + 1, capstyle="round", joinstyle="round")
    elif kind == "warn":
        c.create_polygon(cx, cy - s, cx + s, cy + s * .8, cx - s, cy + s * .8, **kw)
        c.create_text(cx, cy + s * .2, text="!", fill=CARD, font=("Segoe UI", max(8, int(s * 1.1)), "bold"))
    elif kind == "check":
        c.create_line(cx - s * .6, cy, cx - s * .15, cy + s * .5, cx + s * .65, cy - s * .5, **dict(ln, width=w + 1))
    elif kind == "bars":
        for i, hh in enumerate((.5, 1.0, .75)):
            x = cx - s * .9 + i * s * .7
            c.create_rectangle(x, cy + s - 2 * s * hh, x + s * .45, cy + s, **kw)
    elif kind == "clock":
        c.create_oval(cx - s, cy - s, cx + s, cy + s, outline=color, width=w)
        c.create_line(cx, cy, cx, cy - s * .6, **ln)
        c.create_line(cx, cy, cx + s * .5, cy + s * .2, **ln)
    elif kind == "upload":
        c.create_line(cx, cy + s * .2, cx, cy - s * .85, **ln)
        c.create_line(cx - s * .5, cy - s * .3, cx, cy - s * .85, cx + s * .5, cy - s * .3, **ln)
        c.create_line(cx - s, cy + s * .5, cx - s, cy + s * .9, cx + s, cy + s * .9, cx + s, cy + s * .5, **ln)
    elif kind == "play":
        c.create_polygon(cx - s * .6, cy - s, cx - s * .6, cy + s, cx + s * .9, cy, **kw)
    elif kind == "star4":
        c.create_polygon(cx, cy - s, cx + s * .25, cy - s * .25, cx + s, cy, cx + s * .25, cy + s * .25,
                         cx, cy + s, cx - s * .25, cy + s * .25, cx - s, cy, cx - s * .25, cy - s * .25, **kw)
    elif kind == "calendar":
        c.create_rectangle(cx - s, cy - s * .8, cx + s, cy + s, outline=color, width=w)
        c.create_line(cx - s, cy - s * .25, cx + s, cy - s * .25, fill=color, width=w)
        c.create_line(cx - s * .5, cy - s, cx - s * .5, cy - s * .6, **ln)
        c.create_line(cx + s * .5, cy - s, cx + s * .5, cy - s * .6, **ln)


def draw_satellite(c, x, y, k=1.0):
    """A small decorative satellite (body + two solar panels + dish)."""
    panel, edge = "#2a5bb5", "#9bb8f0"
    c.create_polygon(x - 12 * k, y - 3 * k, x - 58 * k, y - 22 * k, x - 66 * k, y - 4 * k, x - 20 * k, y + 14 * k,
                     fill=panel, outline=edge)
    c.create_polygon(x + 12 * k, y + 3 * k, x + 58 * k, y + 22 * k, x + 66 * k, y + 4 * k, x + 20 * k, y - 14 * k,
                     fill=panel, outline=edge)
    c.create_rectangle(x - 13 * k, y - 9 * k, x + 13 * k, y + 9 * k, fill="#cfd8ee", outline="#7f93bd")
    c.create_oval(x - 7 * k, y - 20 * k, x + 7 * k, y - 8 * k, fill="#e6ecf8", outline="#7f93bd")
    c.create_line(x, y - 9 * k, x, y - 26 * k, fill="#9bb8f0", width=2)


# ---------------------------------------------------------------- widgets
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


class Card(tk.Canvas):
    """Rounded 'glass' panel: put children in card.body. Height follows its content."""

    def __init__(self, master, bg=CARD, border=LINE, parent_bg=BG, radius=16, pad=16, stretch=False):
        # initial height > pad so the body window starts inside the canvas (Tk only maps visible windows)
        super().__init__(master, width=10, height=2 * pad + 12, bg=parent_bg, highlightthickness=0)
        self.bgc, self.border, self.radius, self.pad, self.stretch = bg, border, radius, pad, stretch
        self.body = tk.Frame(self, bg=bg)
        self._win = self.create_window(pad, pad, window=self.body, anchor="nw")
        self.bind("<Configure>", self._on_canvas)
        self.body.bind("<Configure>", self._on_body)

    def _on_canvas(self, e):
        kw = {"width": max(e.width - 2 * self.pad, 1)}
        if self.stretch:                       # body fills the card (used by the chart / subsystem cards)
            kw["height"] = max(e.height - 2 * self.pad, 1)
        self.itemconfigure(self._win, **kw)
        self._redraw(e.width, e.height)

    def _on_body(self, e):
        self.configure(height=self.body.winfo_reqheight() + 2 * self.pad)

    def _redraw(self, w=None, h=None):
        w, h = w or self.winfo_width(), h or self.winfo_height()
        self.delete("bg")
        self.create_polygon(round_pts(1, 1, w - 1, h - 1, self.radius), smooth=True, fill=self.bgc,
                            outline=self.border, tags="bg")
        self.tag_lower("bg")

    def recolor(self, bg, border):
        self.bgc, self.border = bg, border
        self.body.configure(bg=bg)
        self._redraw()


class RoundButton(tk.Canvas):
    """Rounded button: filled (primary / secondary) or outlined."""

    def __init__(self, master, text, command, fill=PANEL3, hover=LINE, fg=INK, outline=None, font=FONT_B,
                 padx=18, pady=9, parent_bg=CARD, icon=None, radius=10):
        f = tkfont.Font(font=font)
        iw = 24 if icon else 0
        w, h = f.measure(text) + 2 * padx + iw, f.metrics("linespace") + 2 * pady
        super().__init__(master, width=w, height=h, bg=parent_bg, highlightthickness=0, cursor="hand2")
        self._fill, self._hover, self._outline = fill, hover, outline or fill
        self._poly = self.create_polygon(round_pts(1, 1, w - 1, h - 1, radius), smooth=True, fill=fill,
                                         outline=self._outline, width=1.5 if outline else 1)
        if icon:
            draw_icon(self, icon, padx + 8, h / 2, 8, fg)
        self.create_text(padx + iw + f.measure(text) / 2, h / 2, text=text, fill=fg, font=font)
        self.bind("<Enter>", lambda e: self.itemconfigure(self._poly, fill=self._hover))
        self.bind("<Leave>", lambda e: self.itemconfigure(self._poly, fill=self._fill))
        self.bind("<Button-1>", lambda e: command())


class NavItem(tk.Canvas):
    """Sidebar navigation entry."""

    def __init__(self, master, text, icon, command, width=206, height=46):
        super().__init__(master, width=width, height=height, bg=SIDEBAR, highlightthickness=0, cursor="hand2")
        self.text, self.icon, self.w, self.h = text, icon, width, height
        self.active, self.hover = False, False
        self.bind("<Button-1>", lambda e: command())
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self._draw()

    def _set_hover(self, v):
        self.hover = v
        self._draw()

    def set_active(self, v):
        self.active = v
        self._draw()

    def _draw(self):
        self.delete("all")
        if self.active:
            self.create_polygon(round_pts(1, 1, self.w - 1, self.h - 1, 22), smooth=True, fill=ACTIVE_BG, outline=ACCENT, width=1.5)
        elif self.hover:
            self.create_polygon(round_pts(1, 1, self.w - 1, self.h - 1, 22), smooth=True, fill=CARD2, outline=CARD2)
        draw_icon(self, self.icon, 32, self.h / 2, 10, ACCENT_HOVER if self.active else INK_DIM)
        self.create_text(60, self.h / 2, text=self.text, anchor="w", fill=INK if (self.active or self.hover) else INK_DIM,
                         font=("Segoe UI", 11, "bold" if self.active else "normal"))


class Pill(tk.Canvas):
    """Rounded status badge, e.g. '(dot) Normal'."""

    def __init__(self, master, text="", color=OK, parent_bg=CARD, width=104, height=26, dot=True):
        super().__init__(master, width=width, height=height, bg=parent_bg, highlightthickness=0)
        self.w, self.h, self.dot, self.parent_bg = width, height, dot, parent_bg
        self.set(text, color)

    def set(self, text, color):
        self.delete("all")
        self.create_polygon(round_pts(1, 1, self.w - 1, self.h - 1, self.h / 2 - 1), smooth=True,
                            fill=blend(color, self.parent_bg, .2), outline=blend(color, self.parent_bg, .45))
        if self.dot:
            self.create_oval(14, self.h / 2 - 3, 20, self.h / 2 + 3, fill=color, outline="")
            self.create_text(self.w / 2 + 6, self.h / 2, text=text, fill=color, font=FONT_SMALL_B)
        else:
            self.create_text(self.w / 2, self.h / 2, text=text, fill=color, font=FONT_SMALL_B)


class Bar(tk.Canvas):
    """Thin rounded progress bar."""

    def __init__(self, master, parent_bg=CARD, color=OK, height=6):
        super().__init__(master, width=20, height=height, bg=parent_bg, highlightthickness=0)
        self.frac, self.color, self.hh = 0.0, color, height
        self.bind("<Configure>", lambda e: self._draw())

    def set(self, frac, color):
        self.frac, self.color = max(0.0, min(1.0, frac)), color
        self._draw()

    def _draw(self):
        self.delete("all")
        w, y = max(self.winfo_width(), 20), self.hh / 2
        self.create_line(3, y, w - 3, y, width=self.hh, capstyle="round", fill=TRACK)
        if self.frac > 0.001:
            self.create_line(3, y, 3 + (w - 6) * self.frac, y, width=self.hh, capstyle="round", fill=self.color)


def autowrap(card, label, margin=4):
    """Keep a label's wrap width equal to its card's inner width."""
    card.body.bind("<Configure>", lambda e: label.configure(wraplength=max(160, e.width - margin)), add="+")


class RoundGauge(tk.Canvas):
    def __init__(self, master, size=140, **kw):
        kw.setdefault("bg", CARD)
        super().__init__(master, width=size, height=size, highlightthickness=0, **kw)
        self.size, self.value = size, 0
        self._draw(0, OK)

    def _draw(self, value, color):
        self.delete("all")
        s, pad = self.size, 10
        self.create_oval(pad, pad, s - pad, s - pad, outline=TRACK, width=10)
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
        self.minsize(1120, 700)
        self.entries, self.scales, self._calc = {}, {}, None
        self.det, self.log_raw, self.log_name, self._tl = Detector(), None, "", None
        self._blink_job, self._blink_state = None, False
        self.trees_var, self.cont_var = tk.IntVar(value=200), tk.DoubleVar(value=0.01)
        self.page, self.hist, self.activity, self.processed, self._dots = "dashboard", [], [], 0, []
        rnd = random.Random(7)
        self._stars = [(rnd.random(), rnd.random(), rnd.choice((.8, 1, 1.4))) for _ in range(46)]
        self._build_style()
        self._build_menu()
        self._build_layout()
        self.use_builtin()
        self.load_sample()
        self.bind("<Return>", lambda e: self.run_diagnostics())
        self._tick_clock()
        self.after(350, self._startup_run)

    def _startup_run(self):
        try:
            self._run_manual()          # nominal sample so the dashboard starts populated
        except Exception:
            pass

    # ---------- style / menu ----------
    def _build_style(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure("TNotebook", background=CARD, borderwidth=0, tabmargins=(0, 0, 0, 0))
        st.configure("TNotebook.Tab", background=CARD2, foreground=INK_DIM, padding=(12, 7),
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
        messagebox.showinfo("How the score works", METHOD_TEXT)

    # ---------- layout: sidebar + header + pages ----------
    def _build_layout(self):
        self._build_sidebar()
        main = tk.Frame(self, bg=BG)
        main.pack(side="left", fill="both", expand=True)
        self.hdr = tk.Canvas(main, height=100, bg=BG, highlightthickness=0)
        self.hdr.pack(fill="x")
        self.hdr.bind("<Configure>", lambda e: self._draw_header())
        self.holder = tk.Frame(main, bg=BG)
        self.holder.pack(fill="both", expand=True, padx=26, pady=(4, 18))
        self.pages = {}
        for name, builder in (("dashboard", self._build_dashboard), ("data", self._build_data_page),
                              ("diagnostics", self._build_diag_page), ("about", self._build_about_page)):
            f = tk.Frame(self.holder, bg=BG)
            self.pages[name] = f
            builder(f)
        self.show_page("dashboard")

    def show_page(self, name):
        for f in self.pages.values():
            f.pack_forget()
        self.pages[name].pack(fill="both", expand=True)
        self.page = name
        for n, item in self.nav.items():
            item.set_active(n == name)
        self._draw_header()
        if name == "data":
            self._draw_timeline()
        elif name == "dashboard":
            self._draw_overview()
        elif name == "diagnostics":
            self._update_units()

    def _build_sidebar(self):
        sb = tk.Frame(self, bg=SIDEBAR, width=250)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)
        tk.Frame(self, bg=LINE, width=1).pack(side="left", fill="y")
        logo = tk.Canvas(sb, width=250, height=104, bg=SIDEBAR, highlightthickness=0)
        logo.pack(pady=(26, 0))
        logo.create_oval(68, 24, 182, 84, outline=ACCENT, width=3)
        draw_icon(logo, "star4", 125, 54, 17, ACCENT_HOVER)
        for dx, dy in ((84, 16), (176, 44), (88, 74)):
            logo.create_oval(dx - 1.5, dy - 1.5, dx + 1.5, dy + 1.5, fill=ACCENT_HOVER, outline="")
        tk.Label(sb, text="AstroSense", bg=SIDEBAR, fg=INK, font=("Segoe UI", 20, "bold")).pack()
        tk.Label(sb, text="Monitor \u2022 Analyze \u2022 Stay Ahead", bg=SIDEBAR, fg=INK_DIM,
                 font=FONT_SMALL).pack(pady=(0, 28))
        self.nav = {}
        for name, label, icon in (("dashboard", "Dashboard", "home"), ("data", "Data", "data"),
                                  ("diagnostics", "Diagnostics", "tools"), ("about", "About", "info")):
            it = NavItem(sb, label, icon, lambda n=name: self.show_page(n))
            it.pack(pady=4)
            self.nav[name] = it
        earth = tk.Canvas(sb, height=210, bg=SIDEBAR, highlightthickness=0)
        earth.pack(side="bottom", fill="x")
        earth.bind("<Configure>", lambda e: self._draw_earth(earth))

    def _draw_earth(self, c):
        c.delete("all")
        w, h = max(c.winfo_width(), 250), 210
        for fx, fy, r in self._stars[:14]:
            c.create_oval(fx * w - r, fy * 90 - r, fx * w + r, fy * 90 + r, fill="#5f77a8", outline="")
        c.create_oval(-150, 80, w + 150, h + 420, fill="#0a2550", outline=ACCENT, width=2)
        c.create_oval(-130, 100, w + 130, h + 420, fill="#0c2d5e", outline="#1d4f9e")
        c.create_text(22, 128, anchor="nw", text="Keeping your\nspacecraft healthier\nfor a safer tomorrow.",
                      fill=INK, font=("Segoe UI", 10, "italic"))
        c.create_line(22, 192, 64, 192, fill=ACCENT, width=2)

    def _draw_header(self):
        c = self.hdr
        c.delete("all")
        w, h = max(c.winfo_width(), 900), 100
        for fx, fy, r in self._stars:
            c.create_oval(fx * w - r, fy * h - r, fx * w + r, fy * h + r, fill="#5f77a8", outline="")
        c.create_oval(w - 580, 74, w + 280, 640, outline="#1b4a99", width=2, fill="#081a3c")
        c.create_oval(w - 566, 80, w + 266, 640, outline="#2f78e0", width=2, fill="#0a1f47")
        draw_satellite(c, w - 380, 46, 0.9)
        # date pill
        c.create_polygon(round_pts(w - 262, 22, w - 26, 62, 20), smooth=True, fill=CARD, outline=LINE)
        draw_icon(c, "calendar", w - 238, 42, 8, ACCENT_HOVER, 1)
        c.create_text(w - 142, 42, text="", fill=INK, font=FONT_B, tags="date_txt")
        self._set_date()
        x0 = 30
        if self.page == "dashboard":
            t1 = c.create_text(x0, 38, anchor="w", text="Welcome to ", fill=INK, font=FONT_WELCOME)
            bb = c.bbox(t1)
            x1 = bb[2] if bb else x0 + 190
            t2 = c.create_text(x1, 38, anchor="w", text="AstroSense", fill=ACCENT_HOVER, font=FONT_WELCOME)
            bb = c.bbox(t2)
            x2 = bb[2] if bb else x1 + 170
            draw_icon(c, "star4", x2 + 24, 36, 13, "#ffcb47")
            sub = "Your Spacecraft Diagnostics Console"
        else:
            titles = {"data": ("Data", "Train the model on healthy telemetry and check an uploaded log"),
                      "diagnostics": ("Diagnostics", "Check a single manual sensor reading and read the report"),
                      "about": ("About", "How AstroSense turns telemetry into a health score")}
            title, sub = titles[self.page]
            c.create_text(x0, 38, anchor="w", text=title, fill=INK, font=FONT_TITLE)
        c.create_text(x0, 74, anchor="w", text=sub, fill=INK_DIM, font=FONT_ITALIC)

    def _set_date(self):
        self.hdr.itemconfigure("date_txt", text=time.strftime("%b %d, %Y   |   %H:%M:%S"))

    def _tick_clock(self):
        self._set_date()
        self.after(1000, self._tick_clock)

    # ---------- small builders ----------
    def _card_title(self, body, icon, text, color=ACCENT_HOVER, link=None, cmd=None):
        row = tk.Frame(body, bg=CARD)
        row.pack(fill="x", pady=(0, 8))
        ic = tk.Canvas(row, width=28, height=28, bg=CARD, highlightthickness=0)
        ic.pack(side="left")
        draw_icon(ic, icon, 14, 14, 9, color)
        tk.Label(row, text=text, bg=CARD, fg=INK, font=FONT_H).pack(side="left", padx=(8, 0))
        if link:
            lk = tk.Label(row, text=link + "  \u2192", bg=CARD, fg=ACCENT_HOVER, font=FONT_BOLD, cursor="hand2")
            lk.pack(side="right")
            lk.bind("<Button-1>", lambda e: cmd())
        return row

    def _kpi(self, parent, col, icon, tint, title, value, badge=False, dot=False):
        card = Card(parent, pad=14)
        card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 7, 0 if col == 3 else 7))
        b = card.body
        b.columnconfigure(1, weight=1)
        ic = tk.Canvas(b, width=58, height=58, bg=CARD, highlightthickness=0)
        ic.grid(row=0, column=0, rowspan=2, padx=(0, 12))
        ic.create_oval(2, 2, 56, 56, fill=blend(tint, CARD, .22), outline=blend(tint, CARD, .55))
        draw_icon(ic, icon, 29, 29, 13, tint)
        tk.Label(b, text=title, bg=CARD, fg=INK, font=FONT_B).grid(row=0, column=1, sticky="sw")
        vf = tk.Frame(b, bg=CARD)
        vf.grid(row=1, column=1, sticky="nw")
        val = tk.Label(vf, text=value, bg=CARD, fg=INK, font=("Segoe UI", 22, "bold"))
        val.pack(side="left")
        extra = None
        if badge:
            extra = Pill(vf, "Healthy", OK, width=92, dot=False)
            extra.pack(side="left", padx=(14, 0))
        elif dot:
            extra = tk.Canvas(vf, width=12, height=12, bg=CARD, highlightthickness=0)
            extra.pack(side="left", padx=(10, 0), pady=(8, 0))
            self._dots.append(extra)
        bar = Bar(b, color=tint)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        return val, bar, extra

    # ---------- Dashboard ----------
    def _build_dashboard(self, page):
        kp = tk.Frame(page, bg=BG)
        kp.pack(fill="x")
        for i in range(4):
            kp.columnconfigure(i, weight=1, uniform="kpi")
        self.k_health, self.k_health_bar, self.k_badge = self._kpi(kp, 0, "pulse", ACCENT, "Spacecraft Health", "\u2014", badge=True)
        self.k_data, self.k_data_bar, _ = self._kpi(kp, 1, "data", PURPLE, "Data Processed", "0")
        self.k_alert, self.k_alert_bar, _ = self._kpi(kp, 2, "warn", WARN, "Alerts", "0")
        self.k_status, self.k_status_bar, self.k_dot = self._kpi(kp, 3, "shield", OK, "System Status", "\u2014", dot=True)

        mid = tk.Frame(page, bg=BG)
        mid.pack(fill="both", expand=True, pady=(12, 0))
        mid.columnconfigure(0, weight=2, uniform="m")
        mid.columnconfigure(1, weight=1, uniform="m")
        mid.rowconfigure(0, weight=1)
        ch = Card(mid, stretch=True)
        ch.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        self._card_title(ch.body, "bars", "Telemetry Overview", link="View Details",
                         cmd=lambda: self.show_page("diagnostics"))
        self.ov = tk.Canvas(ch.body, height=OV_H, bg=CARD, highlightthickness=0)
        self.ov.pack(fill="both", expand=True)
        self.ov.bind("<Configure>", lambda e: self._draw_overview())
        lg = tk.Frame(ch.body, bg=CARD)
        lg.pack(pady=(4, 0))
        d = tk.Canvas(lg, width=10, height=10, bg=CARD, highlightthickness=0)
        d.create_oval(0, 0, 10, 10, fill=ACCENT, outline="")
        d.pack(side="left", padx=(0, 6))
        tk.Label(lg, text="Health Score", bg=CARD, fg=INK_DIM, font=FONT_SMALL).pack(side="left")

        sc = Card(mid, stretch=True)
        sc.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        self._card_title(sc.body, "chip", "Subsystem Status")
        self.sub_pills = {}
        for gid, gname, _ in GROUPS:
            row = tk.Frame(sc.body, bg=CARD2)
            row.pack(fill="x", pady=3)
            ic = tk.Canvas(row, width=34, height=34, bg=CARD2, highlightthickness=0)
            ic.pack(side="left", padx=(8, 4), pady=3)
            draw_icon(ic, GROUP_ICON[gid], 17, 17, 9, ACCENT_HOVER)
            tk.Label(row, text=gname, bg=CARD2, fg=INK, font=FONT_B).pack(side="left", padx=6)
            pill = Pill(row, "Normal", OK, parent_bg=CARD2, width=96)
            pill.pack(side="right", padx=8)
            self.sub_pills[gid] = pill

        bot = tk.Frame(page, bg=BG)
        bot.pack(fill="x", pady=(12, 0))
        bot.columnconfigure(0, weight=1, uniform="b")
        bot.columnconfigure(1, weight=1, uniform="b")
        qa = Card(bot)
        qa.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        self._card_title(qa.body, "bolt", "Quick Actions")
        br = tk.Frame(qa.body, bg=CARD)
        br.pack(fill="x", pady=(6, 0))
        RoundButton(br, "Upload Telemetry Log", self.upload_log, fill=ACCENT, hover=ACCENT_HOVER, fg="white",
                    font=FONT_BOLD, padx=22, pady=12, icon="upload").pack(side="left", padx=(0, 12))
        RoundButton(br, "Run Diagnostics", self.run_diagnostics, fill=CARD, hover=CARD2, fg=INK, outline=ACCENT,
                    font=FONT_BOLD, padx=22, pady=12, icon="play").pack(side="left")
        ra = Card(bot)
        ra.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        self._card_title(ra.body, "clock", "Recent Activity", link="View All", cmd=self._show_activity)
        self.act_box = tk.Frame(ra.body, bg=CARD)
        self.act_box.pack(fill="x")
        self._render_activity()

    def _log_event(self, text, kind="ok"):
        self.activity.insert(0, (time.strftime("%H:%M:%S"), text, kind))
        self.activity = self.activity[:60]
        self._render_activity()

    def _render_activity(self):
        for w in self.act_box.winfo_children():
            w.destroy()
        if not self.activity:
            tk.Label(self.act_box, text="No activity yet.", bg=CARD, fg=INK_DIM, font=FONT_B).pack(anchor="w")
        for stamp, text, kind in self.activity[:3]:
            col, icon = {"ok": (OK, "check"), "warn": (WARN, "warn")}.get(kind, (ACCENT_HOVER, "clock"))
            row = tk.Frame(self.act_box, bg=CARD)
            row.pack(fill="x", pady=3)
            ic = tk.Canvas(row, width=30, height=30, bg=CARD, highlightthickness=0)
            ic.pack(side="left", padx=(0, 10))
            ic.create_oval(1, 1, 29, 29, fill=blend(col, CARD, .25), outline="")
            draw_icon(ic, icon, 15, 15, 7, col, 2)
            tf = tk.Frame(row, bg=CARD)
            tf.pack(side="left", fill="x")
            tk.Label(tf, text=text, bg=CARD, fg=INK, font=FONT_B, wraplength=420, justify="left").pack(anchor="w")
            tk.Label(tf, text=stamp, bg=CARD, fg=INK_DIM, font=FONT_SMALL).pack(anchor="w")

    def _show_activity(self):
        win = tk.Toplevel(self, bg=BG)
        win.title("Activity log")
        win.geometry("640x420")
        txt = tk.Text(win, bg=CARD, fg=INK, font=FONT_B, wrap="word", relief="flat", padx=14, pady=10)
        txt.pack(fill="both", expand=True)
        for stamp, text, _ in self.activity:
            txt.insert("end", f"{stamp}   {text}\n\n")
        txt.configure(state="disabled")

    def _draw_overview(self):
        c = self.ov
        c.delete("all")
        w, h = max(c.winfo_width(), 300), max(c.winfo_height(), 160)
        pl, pr, pt, pb = 44, 36, 14, 30
        y_of = lambda v: pt + (h - pt - pb) * (1 - v / 100)
        pts = self.hist
        n = len(pts)
        x_of = lambda i: pl + (w - pl - pr) * (i / (n - 1) if n > 1 else 0.5)
        if n >= 2:
            area = [x_of(0), y_of(0)]
            for i, (_, v) in enumerate(pts):
                area += [x_of(i), y_of(v)]
            area += [x_of(n - 1), y_of(0)]
            c.create_polygon(*area, fill=blend(ACCENT, CARD, .3), outline="")
        for v in (0, 25, 50, 75, 100):
            c.create_line(pl, y_of(v), w - pr, y_of(v), fill=LINE)
            c.create_text(pl - 8, y_of(v), text=str(v), fill=INK_DIM, font=FONT_SMALL, anchor="e")
        if not pts:
            c.create_text(w / 2, h / 2, text="Run diagnostics to build the health history", fill=INK_DIM, font=FONT_B)
            return
        step = max(1, -(-n // 6))
        for i, (stamp, v) in enumerate(pts):
            if i % step == 0 or i == n - 1:
                c.create_line(x_of(i), pt, x_of(i), h - pb, fill=LINE)
                c.create_text(x_of(i), h - pb + 14, text=stamp, fill=INK_DIM, font=FONT_SMALL)
        if n >= 2:
            line = []
            for i, (_, v) in enumerate(pts):
                line += [x_of(i), y_of(v)]
            c.create_line(*line, fill=ACCENT, width=2.5, smooth=True)
        for i, (_, v) in enumerate(pts):
            c.create_oval(x_of(i) - 4, y_of(v) - 4, x_of(i) + 4, y_of(v) + 4, fill=ACCENT_HOVER, outline=CARD, width=2)

    # ---------- Diagnostics (manual sensor tabs + report) ----------
    def _build_diag_page(self, page):
        page.columnconfigure(0, weight=6, uniform="d")
        page.columnconfigure(1, weight=5, uniform="d")
        page.rowconfigure(0, weight=1)
        left = Card(page)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        b = left.body
        self._card_title(b, "tools", "Manual Sensor Reading")
        hint = tk.Label(b, text="Pick a subsystem tab and enter readings (type a value or drag the slider). "
                                "Blank fields are treated as nominal.", bg=CARD, fg=INK_DIM, font=FONT_SMALL,
                        justify="left", wraplength=520)
        hint.pack(anchor="w", pady=(0, 6))
        autowrap(left, hint)
        self.nb = ttk.Notebook(b)
        self.nb.pack(fill="x")
        self.nb.bind("<<NotebookTabChanged>>", self._update_units)
        for gid, gname, _ in GROUPS:
            tab = tk.Frame(self.nb, bg=CARD)
            self.nb.add(tab, text=TAB_LABELS.get(gid, gname.split(" ")[0]))
            for f in [f for f in FIELDS if f["group"] == gid]:
                self._build_field_row(tab, f)
        br = tk.Frame(b, bg=CARD)
        br.pack(fill="x", pady=(14, 6))
        RoundButton(br, "Run Diagnostics", self.run_diagnostics, fill=ACCENT, hover=ACCENT_HOVER, fg="white",
                    font=FONT_BOLD, padx=18, pady=9, icon="play").pack(side="left", padx=(0, 10))
        for txt, cmd in (("Nominal Sample", self.load_sample), ("Simulate Fault", self.load_fault), ("Clear", self.clear_all),
                         ("Upload Log...", self.upload_log)):
            RoundButton(br, txt, cmd, fill=PANEL3, hover=LINE, padx=14, pady=9).pack(side="left", padx=(0, 8))
        tk.Frame(b, bg=LINE, height=1).pack(fill="x", pady=(6, 8))
        ur = tk.Frame(b, bg=CARD)
        ur.pack(fill="x")
        tk.Label(ur, text="UNITS", bg=CARD, fg=ACCENT_HOVER, font=FONT_CAP).pack(side="left", anchor="n", padx=(0, 10), pady=(2, 0))
        self.units_lbl = tk.Label(ur, bg=CARD, fg=INK, font=FONT_SMALL, anchor="w", justify="left", wraplength=480)
        self.units_lbl.pack(side="left", fill="x", expand=True)
        ur.bind("<Configure>", lambda e: self.units_lbl.configure(wraplength=max(200, e.width - 70)))
        tk.Label(b, text="Press Enter to run diagnostics.", bg=CARD, fg=INK_DIM, font=FONT_SMALL).pack(anchor="w", pady=(8, 0))
        self._update_units()

        right = tk.Frame(page, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        self._build_result_panel(right)

    def _update_units(self, _=None):
        i = self.nb.index(self.nb.select())
        self.units_lbl.configure(text=units_text(GROUPS[i][0]))

    def _build_field_row(self, parent, f):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", padx=2, pady=(12, 0))
        top = tk.Frame(row, bg=CARD)
        top.pack(fill="x")
        lbl = tk.Label(top, text=f["label"], bg=CARD, fg=INK, font=FONT_BOLD, cursor="question_arrow")
        lbl.pack(side="left")
        Tooltip(lbl, f'Range: {f["min"]}\u2013{f["max"]} {f["unit"]}\n{f["src"][:110]}')
        var = tk.StringVar()
        tk.Entry(top, textvariable=var, width=10, bg=BG, fg=INK, insertbackground=INK, relief="flat",
                 highlightthickness=1, highlightbackground=LINE, highlightcolor=ACCENT, font=FONT_MONO,
                 justify="right").pack(side="right", ipady=2)
        tk.Label(top, text=f["unit"], bg=CARD, fg=INK_DIM, font=FONT_SMALL).pack(side="right", padx=(4, 8))
        tk.Label(row, text=f'Nominal {f["min"]}\u2013{f["max"]} {f["unit"]}', bg=CARD, fg=INK_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        lo, hi = f["hard_min"], f["hard_max"]
        scale = tk.Scale(row, from_=lo, to=hi, orient="horizontal", resolution=(hi - lo) / 200, showvalue=False,
                         bg=ACCENT, troughcolor=PANEL3, highlightthickness=0, bd=0, sliderrelief="flat",
                         sliderlength=22, width=14, length=300, activebackground=ACCENT_HOVER)
        scale.pack(fill="x", pady=(4, 0))
        tk.Frame(parent, bg=LINE, height=1).pack(fill="x", padx=2, pady=(10, 0))
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

    def _build_result_panel(self, right):
        top = tk.Frame(right, bg=BG)
        top.pack(fill="x")
        self.status_card = Card(top)
        self.status_card.pack(side="left", fill="both", expand=True, padx=(0, 10))
        sb = self.status_card.body
        self.status_dot = tk.Canvas(sb, width=14, height=14, bg=CARD, highlightthickness=0)
        self.status_dot.pack(anchor="w", pady=(4, 0))
        self._dots.append(self.status_dot)
        self.status_title = tk.Label(sb, text="Awaiting data", bg=CARD, fg=INK, font=("Segoe UI", 14, "bold"))
        self.status_title.pack(anchor="w", pady=(6, 0))
        self.status_sub = tk.Label(sb, text="Enter a reading or upload a log, then run diagnostics.",
                                   bg=CARD, fg=INK_DIM, font=FONT_B, wraplength=260, justify="left")
        self.status_sub.pack(anchor="w", pady=(2, 4))
        autowrap(self.status_card, self.status_sub)
        gc = Card(top)
        gc.pack(side="left", fill="y")
        self.gauge = RoundGauge(gc.body, size=130, bg=CARD)
        self.gauge.pack(pady=(0, 6))
        RoundButton(gc.body, "Show calculation", self._show_calc, fill=PANEL3, hover=LINE, font=FONT_SMALL,
                    padx=12, pady=5).pack()

        stats = tk.Frame(right, bg=BG)
        stats.pack(fill="x", pady=(10, 0))
        for i in range(3):
            stats.columnconfigure(i, weight=1, uniform="st")
        self.stat_health = self._stat_box(stats, 0, "System status", "\u2014")
        self.stat_flags = self._stat_box(stats, 1, "Flagged records", "0")
        self.stat_quality = self._stat_box(stats, 2, "Input quality", "\u2014")

        cc = Card(right)
        cc.pack(fill="x", pady=(10, 0))
        self._card_title(cc.body, "warn", "Warning Checklist", color=WARN)
        self.chk_box = tk.Frame(cc.body, bg=CARD)
        self.chk_box.pack(fill="x")
        tk.Label(self.chk_box, text="No analysis yet.", bg=CARD, fg=INK_DIM, font=FONT_B).pack(anchor="w")

        self.insight_card = Card(right, bg=INSIGHT_BG, border=ACCENT)
        self.insight_card.pack(fill="x", pady=(10, 0))
        self.insight = tk.Label(self.insight_card.body, text="No readings screened yet.", bg=INSIGHT_BG,
                                fg=INK_DIM, font=FONT_B, wraplength=440, justify="left")
        self.insight.pack(anchor="w")
        autowrap(self.insight_card, self.insight)
        self._set_dot(OK)

    def _stat_box(self, parent, col, title, value):
        card = Card(parent, pad=12)
        card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 5, 0 if col == 2 else 5))
        tk.Label(card.body, text=title.upper(), bg=CARD, fg=INK_DIM, font=FONT_CAP).pack(anchor="w")
        v = tk.Label(card.body, text=value, bg=CARD, fg=INK, font=("Segoe UI", 14, "bold"))
        v.pack(anchor="w", pady=(2, 0))
        return v

    def _set_dot(self, color):
        for d in self._dots:
            d.delete("all")
            d.create_oval(1, 1, int(d["width"]) - 1, int(d["height"]) - 1, fill=color, outline="")

    # ---------- Data page ----------
    def _build_data_page(self, page):
        page.columnconfigure(0, weight=1, uniform="p")
        page.columnconfigure(1, weight=1, uniform="p")
        c1 = Card(page)
        c1.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        b = c1.body
        self._card_title(b, "data", "1 \u00B7 Healthy baseline (model training)")
        self.base_lbl = tk.Label(b, bg=CARD, fg=INK_DIM, font=FONT_SMALL, wraplength=420, justify="left")
        self.base_lbl.pack(anchor="w", pady=(0, 6))
        autowrap(c1, self.base_lbl)
        r = tk.Frame(b, bg=CARD)
        r.pack(fill="x", pady=4)
        for txt, cmd in (("Load healthy CSV...", self.load_baseline_csv), ("Use built-in baseline", self.use_builtin),
                         ("Retrain", self.retrain)):
            RoundButton(r, txt, cmd, fill=PANEL3, hover=LINE, padx=12, pady=7).pack(side="left", padx=(0, 6))
        r = tk.Frame(b, bg=CARD)
        r.pack(fill="x", pady=(8, 0))
        tk.Label(r, text="Trees", bg=CARD, fg=INK_DIM, font=FONT_SMALL).pack(side="left")
        ttk.Spinbox(r, from_=50, to=500, increment=50, textvariable=self.trees_var, width=6,
                    style="Dark.TSpinbox").pack(side="left", padx=(6, 18))
        tk.Label(r, text="Expected anomalies", bg=CARD, fg=INK_DIM, font=FONT_SMALL).pack(side="left")
        ttk.Spinbox(r, from_=0.001, to=0.2, increment=0.005, format="%.3f", textvariable=self.cont_var, width=7,
                    style="Dark.TSpinbox").pack(side="left", padx=6)

        c2 = Card(page)
        c2.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        b = c2.body
        self._card_title(b, "upload", "2 \u00B7 Telemetry log to check")
        self.log_lbl = tk.Label(b, text="No log uploaded yet.", bg=CARD, fg=INK_DIM, font=FONT_SMALL,
                                wraplength=420, justify="left")
        self.log_lbl.pack(anchor="w", pady=(0, 6))
        autowrap(c2, self.log_lbl)
        r = tk.Frame(b, bg=CARD)
        r.pack(fill="x", pady=4)
        RoundButton(r, "Upload log CSV...", self.upload_log, fill=ACCENT, hover=ACCENT_HOVER, fg="white",
                    font=FONT_BOLD, padx=16, pady=8, icon="upload").pack(side="left", padx=(0, 8))
        RoundButton(r, "Run on log", self._run_log, fill=PANEL3, hover=LINE, padx=14, pady=8).pack(side="left")

        c3 = Card(page)
        c3.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self._card_title(c3.body, "bars", "3 \u00B7 Health score over time")
        self.tl = tk.Canvas(c3.body, bg="#08122a", height=250, highlightthickness=1, highlightbackground=LINE)
        self.tl.pack(fill="x", pady=(2, 0))
        self.tl.bind("<Configure>", lambda e: self._draw_timeline())

    # ---------- About page ----------
    def _build_about_page(self, page):
        card = Card(page)
        card.pack(fill="x")
        b = card.body
        self._card_title(b, "info", "How the score works")
        tk.Label(b, text="Machine Learning for Autonomous Spacecraft Diagnostics", bg=CARD, fg=ACCENT_HOVER,
                 font=FONT_BOLD).pack(anchor="w", pady=(0, 8))
        msg = tk.Label(b, text=METHOD_TEXT, bg=CARD, fg=INK, font=FONT_B, justify="left", wraplength=900)
        msg.pack(anchor="w")
        autowrap(card, msg)
        RoundButton(b, "Ranges, equations & sources", self._show_sources, fill=ACCENT, hover=ACCENT_HOVER,
                    fg="white", font=FONT_BOLD, padx=18, pady=9).pack(anchor="w", pady=(16, 0))

    # ---------- blink / helpers ----------
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

    def load_fault(self):
        self.load_sample()
        for k, v in FAULT_OVERRIDES.items():
            self.entries[k].set(str(v))
        self.nb.select(0)

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
        self._log_event(f"Model trained on {len(df)} healthy records.", "info")
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
        self._log_event(f"Log uploaded: {self.log_name}", "info")
        self._run_log()

    def run_diagnostics(self):
        """Data page -> uploaded log; Diagnostics page -> manual reading; Dashboard -> log if one is loaded."""
        page = getattr(self, "page", "dashboard")
        if page == "data" or (page == "dashboard" and self.log_raw is not None):
            self._run_log()
        else:
            self._run_manual()

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
        self._show(df, f"{dropped} rows dropped", t0, t1, labels, self.log_name,
                   clean_frac=len(df) / max(len(self.log_raw), 1))

    def _run_manual(self):
        row, filled = {}, 0
        for c in self.det.cols:
            v = self._parse(c)
            filled += v is not None
            row[c] = v if v is not None else float(self.det.mean[c])   # blank field -> assumed nominal
        if filled == 0:
            messagebox.showinfo("No readings", "Enter at least one reading first.")
            return
        self._show(pd.DataFrame([row]), f"{filled}/{len(self.det.cols)} fields", None, None, ["this reading"],
                   clean_frac=filled / len(self.det.cols))

    def _show(self, df, quality, t0, t1, labels, source="this reading", clean_frac=1.0):
        health, flagged, z = self.det.score(df)
        n, frac = len(df), float(flagged.mean())
        overall = int(round(health.mean() if n < 30 else health.mean() * (1 - frac)))
        self._calc = self.det.report(df, flagged, overall, source)
        is_anom = bool(flagged.any()) if n < 30 else frac > 0.03
        items = self.det.explain(flagged, z) if flagged.any() else []
        color = OK if overall >= 80 else (WARN if overall >= 55 else DANGER)
        nflag = int(flagged.sum())
        self.processed += n

        # ---- Diagnostics page: status card, gauge, stats
        self.gauge.animate_to(overall, color)
        bg = DANGER_BG if is_anom else CARD
        self.status_card.recolor(bg, DANGER if is_anom else OK)
        for w in (self.status_dot, self.status_title, self.status_sub):
            w.configure(bg=bg)
        self.status_title.configure(text="Anomaly detected" if is_anom else "No anomaly detected", fg=DANGER if is_anom else OK)
        self.status_sub.configure(text=f"{nflag} of {n} record(s) deviate from the learned healthy pattern."
                                  if is_anom else "Data matches the learned healthy pattern.")
        self._stop_blink()
        self._start_blink() if is_anom else self._set_dot(OK)
        self.stat_health.configure(text="ANOMALOUS" if is_anom else "NOMINAL", fg=DANGER if is_anom else OK)
        self.stat_flags.configure(text=f"{nflag} / {n}")
        self.stat_quality.configure(text=quality)

        if t0 is not None:
            self._tl = (np.asarray(health), np.asarray(flagged), t0, t1)
            self._draw_timeline()

        # ---- warning checklist
        for w in self.chk_box.winfo_children():
            w.destroy()
        if not items:
            tk.Label(self.chk_box, text="No warnings \u2014 all systems nominal.", bg=CARD, fg=OK, font=FONT_B).pack(anchor="w")
        for col, cnt, mz in items[:5]:
            c = DANGER if abs(mz) >= 4 else WARN
            r = tk.Frame(self.chk_box, bg=DANGER_BG)
            r.pack(fill="x", pady=3)
            tk.Frame(r, bg=c, width=3).pack(side="left", fill="y")
            lab = tk.Label(r, bg=DANGER_BG, fg=INK, font=FONT_SMALL, anchor="w", padx=8, pady=5, cursor="hand2",
                           text=f"\u2610  {warning_text(col, mz)}   \u00B7  {cnt} record(s), {mz:+.1f}\u03C3")
            lab.pack(side="left", fill="x")
            lab.bind("<Button-1>", lambda e, l=lab: l.configure(
                text=("\u2611" if l.cget("text")[0] == "\u2610" else "\u2610") + l.cget("text")[1:]))
        if len(items) > 5:
            tk.Label(self.chk_box, text=f"\u2026and {len(items) - 5} more", bg=CARD, fg=INK_DIM, font=FONT_SMALL).pack(anchor="w")

        # ---- subsystem status (dashboard)
        gstate = {g: 0 for g, _, _ in GROUPS}
        for col, cnt, mz in items:
            if col in FBY:
                g = FBY[col]["group"]
                gstate[g] = max(gstate[g], 2 if abs(mz) >= 4 else 1)
        for gid, pill in self.sub_pills.items():
            s = gstate[gid]
            pill.set(*(("Normal", OK) if s == 0 else (("Watch", WARN) if s == 1 else ("Fault", DANGER))))

        # ---- insight
        if is_anom and items:
            first = labels[int(np.flatnonzero(flagged)[0])]
            text = (f"{warning_text(items[0][0], items[0][2])} is the strongest signal. First flagged at "
                    f"{first}. Inspect this subsystem before proceeding.")
        elif is_anom:
            text = "Unusual combination of readings detected; no single sensor stands out."
        else:
            text = f"{n} record(s) screened against the healthy baseline; nothing unusual found."
        self.insight_card.recolor(INSIGHT_BG, DANGER if is_anom else ACCENT)
        self.insight.configure(text=text, fg=INK)

        # ---- dashboard KPIs, chart, activity
        self.k_health.configure(text=f"{overall}%")
        self.k_badge.set("Healthy" if overall >= 80 else ("Watch" if overall >= 55 else "Critical"), color)
        self.k_health_bar.set(overall / 100, color)
        self.k_data.configure(text=f"{self.processed:,}")
        self.k_data_bar.set(clean_frac, PURPLE)
        self.k_alert.configure(text=str(nflag), fg=(DANGER if is_anom else WARN) if nflag else INK)
        self.k_alert_bar.set(nflag / max(n, 1), DANGER if is_anom else WARN)
        self.k_status.configure(text="Anomalous" if is_anom else "Nominal", fg=DANGER if is_anom else OK)
        self.k_status_bar.set(overall / 100, color)
        self.hist = (self.hist + [(time.strftime("%H:%M:%S"), overall)])[-12:]
        self._draw_overview()
        if is_anom and items:
            self._log_event(f"{nflag} of {n} record(s) flagged. {text.split(' is the strongest')[0]} is the strongest signal.", "warn")
        elif is_anom:
            self._log_event(f"{nflag} of {n} record(s) flagged as unusual.", "warn")
        else:
            self._log_event("Diagnostics completed successfully.", "ok")

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
            self._log_event(f"OPS-SAT-AD evaluated: precision {p:.2f}, recall {r:.2f}, F1 {f1:.2f}", "info")
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

