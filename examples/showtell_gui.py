#!/usr/bin/env python3
"""
PBC Show & Tell demo GUI — ICIP 2026, Tampere.

One-window, big-font demo for walk-up attendees:
  1. PROTECT   — encode PBC into the loaded image, save protected PNG
  2. DRAW      — scribble on the protected image right in the window, or
                 open it in Preview/Paint and save (copies are picked up)
  3. VERIFY    — reload the file, verify, show GREEN/RED overlay + verdict

Dependencies: numpy, Pillow, tkinter (stdlib). No GPU, no network.
MIT License — Francois Legare, 2026.

Run:  python examples/showtell_gui.py            (add --demo to start the
      auto-demo loop; any click or key pauses it where it is — "Resume demo"
      continues from that step, "Demo from start" restarts the round;
      --pace 1.3 gives every explainer 30% more time)
"""

import os
import sys
import subprocess
import platform
import tempfile
import glob
import math
import random
import types
import time
import threading
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
try:
    import resource               # peak-RAM readout (Unix only)
except ImportError:
    resource = None
import tkinter as tk
from tkinter import filedialog, messagebox

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbc import OpCode, generate_originator_id, compute_grid
from pbc.encoder import encode, encode_region
from pbc.decoder import verify, TileStatus, BlockStatus, extract_edit_ledger
from pbc.visualizer import generate_overlay

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(HERE, "img", "leo.jpg")
WORK_PNG = os.path.join(tempfile.gettempdir(), "pbc_demo_protected.png")

# ICIP 2026 / IEEE brand blues (2026.ieeeicip.org) + semantic tile colours
NAVY, BLUE, ICE = "#002855", "#00629B", "#E4ECF4"
GREEN, RED, YELLOW, GRAY = "#27AE60", "#E74C3C", "#F2C94C", "#9AA5B1"
PANE_BG = "#001B3A"
BTN_OFF_BG, BTN_OFF_FG = "#12365E", "#5C7896"   # disabled: dim but readable
ALL_STEPS = {"load", "protect", "edit", "preview", "verify", "reset"}
BRUSH, BRUSH_PX = "#FF00FF", 14   # in-window pen: colour, on-screen width
BRUSH_EDITOR = "#00C2FF"          # PBC-aware editor pen (re-encoded + logged)
PEN_LEGEND = [(BRUSH, "■ MAGENTA = raw tamper → RED"),
              (BRUSH_EDITOR, "■ CYAN = PBC-aware edit, logged → YELLOW"),
              ("white", "⧉ WHITE BOX = copied tile, not logged → YELLOW")]
ORIGINATOR, EDITOR_ID = "icip2026-showtell", "ShowTell-Editor"   # self-asserted names
KNOWN_IDS = {generate_originator_id(n): n for n in (ORIGINATOR, EDITOR_ID)}
# where an external editor may drop "pbc_demo_protected copy.png" — Preview's
# Save dialog defaults to the last folder used, not the hidden temp dir
WATCH_DIRS = [os.path.dirname(WORK_PNG), HERE, os.path.dirname(HERE),
              os.path.dirname(os.path.dirname(HERE))] + [
    os.path.expanduser(f"~/{d}") for d in ("Desktop", "Downloads", "Documents", "Pictures")]
MAX_SIDE = 1024   # downscale big photos so encode/verify stay ~1 s
# auto-demo pacing: each explainer stays up long enough to read AND absorb it
READING_WPM = 90         # slow, deliberate reading for a TV watched from a few metres
ABSORB_MS = 5000         # extra time to look at the image and make sense of it
CURSOR_HOT = 28          # arrow tip position inside the demo cursor image
# Slide 4 comparison, measured 13 Sep 2026 on the demo laptop (MacBookPro11,5,
# i7-4870HQ, CPU only), 978×678 photo: the vectorized PBC implementation (1 core)
# vs TrustMark "Q" (Adobe; PyTorch 2.2.2, 1 thread). Fixed values, not re-measured live.
# (label, unit, PBC, TrustMark, lower is better, grows in real time)
RUN_COMPARE = [("Embed", "s", 0.029, 0.494, True, True),
               ("Check", "s", 0.047, 0.168, True, True),
               ("Start-up", "s", 0.14, 4.0, True, True),
               ("Model weights", "MB", 0, 62, True, False),
               ("Peak memory", "MB", 71, 709, True, False),
               ("Image quality", "dB", 51.2, 43.4, False, False)]
RUN_COMPARE_TEXT = [("Software", "NumPy + Pillow", "PyTorch + Lightning"),
                    ("Tells you", "which tiles changed", "watermark present?")]
DEMO_SECTIONS = ["Intro", "Protect", "One chain per tile", "What it takes", "Edit",
                 "Verify", "Verdict", "Edit Ledger", "Intact tile", "Tampered tile",
                 "Copied tile"]
PREVIEW_W, PREVIEW_H = 560, 302   # fixed image boxes; leaves room for the explainer bar


def open_in_editor(path):
    """Open file in an image editor (Paint on Windows, default app elsewhere)."""
    system = platform.system()
    if system == "Windows":
        subprocess.Popen(["mspaint", path])
    elif system == "Darwin":
        subprocess.Popen(["open", "-a", "Preview", path])
    else:
        for cand in ("kolourpaint", "gimp", "xdg-open"):
            try:
                subprocess.Popen([cand, path])
                return
            except FileNotFoundError:
                continue


def machine_name():
    """(model, cpu) for the resource card, e.g. ('MacBookPro11,5', 'Intel Core i7-4870HQ @ 2.50GHz')."""
    model, cpu = platform.machine(), platform.processor()
    if platform.system() == "Darwin":
        try:
            q = lambda k: subprocess.run(["sysctl", "-n", k], capture_output=True,
                                         text=True, timeout=2).stdout.strip()
            model, cpu = q("hw.model") or model, q("machdep.cpu.brand_string") or cpu
        except (OSError, subprocess.SubprocessError):
            pass
    for junk in ("(R)", "(TM)", " CPU"):
        cpu = cpu.replace(junk, "")
    return model, " ".join(cpu.split())


LABEL_FONTS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",   # macOS
               "/System/Library/Fonts/SFNS.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]


def label_font(size):
    """(font, ascii_only): a bold TrueType font that has × and →, else Pillow's
    built-in font, whose missing glyphs would render as boxes."""
    for name in LABEL_FONTS:
        try:
            return ImageFont.truetype(name, size), False
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size), True
    except TypeError:                    # Pillow < 10.1
        return ImageFont.load_default(), True


def peak_ram_mb():
    """Peak resident memory of this whole process, or None where unsupported."""
    if resource is None:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if platform.system() == "Darwin" else rss / 2**10   # bytes vs KiB


class Demo(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pixel Block Chain — live demo (ICIP 2026)")
        self.configure(bg=NAVY)
        self.geometry("1420x820")
        self.original = None          # np.ndarray, loaded photo
        self.protected = None         # np.ndarray, encoded
        self.work_mtime = None        # newest saved-file mtime seen (live watcher)
        self.verify_path = None       # file VERIFY reads back from disk
        self.edit_img = None          # PIL image the DRAW step paints on
        self.dirty = False            # in-window strokes not saved yet
        self.drawing = False
        self.last_pt = None
        self.demo_on, self.demo_job, self.demo_gen = False, None, None
        self.demo_delay, self.demo_bar = 0, None    # pending pause + card, kept for resume
        self.demo_section = 0         # current slide (index into DEMO_SECTIONS)
        self.demo_ff = 0              # replay quickly until this slide (jump target)
        self.pie_span = False         # one countdown spans several yields (slide actions)
        self.grid_job = None          # pending grid-overlay fade
        self.compare_job = None       # pending slide-4 comparison animation frame
        self.compare_fonts = self.compare_bg = None
        self.pen_compliant = False    # False: raw tamper pen, True: PBC-aware editor pen
        self.mask_img = self.mask_owner = None      # PBC-aware strokes, per edit_img
        self.verify_result = self.verdict_thumb = None
        self.base_oid = None          # original encoder's oid in the verified image
        self.ledger_pick = None       # tile whose ledger VERIFY opened automatically
        self.copy_target = None       # tile the demo's copy-paste attack pasted onto
        self.pace = 1.0               # --pace multiplier on auto-demo reading times
        self.explain_mono = False
        self.busy = False             # PROTECT / VERIFY running in a worker thread
        self.est_s = {"protect": 0.5, "verify": 0.3}   # sweep pace, re-measured each run
        self.cursor_pos = None        # demo cursor tip, screen coordinates
        self.last_result = None       # GridResult of the latest detection check
        self.last_encode_s = 0.0
        self.machine = machine_name()

        big = ("Segoe UI", 16, "bold")
        huge = ("Segoe UI", 22, "bold")

        tk.Label(self, text="Tamper with our image — we'll show you exactly where.",
                 font=huge, fg="white", bg=NAVY).pack(pady=(10, 2))
        self.status = tk.Label(self, text="Load a photo (or use the default) to start.",
                               font=("Segoe UI", 14), fg=ICE, bg=NAVY)
        self.status.pack()

        btns = tk.Frame(self, bg=NAVY)
        btns.pack(pady=10)
        # tk.Label, not tk.Button: macOS Aqua buttons ignore bg (white on gray)
        self.buttons = {}

        def mk(key, txt, cmd, parent=btns, font=big, pady=10):
            b = tk.Label(parent, text=txt, font=font, padx=18, pady=pady,
                         highlightthickness=3, highlightbackground=NAVY)   # border for _flash
            b.bind("<Button-1>", lambda e: (self._discard_demo(), cmd())
                   if b.enabled and not (self.demo_on or self.busy) else None)
            b.pack(side="left", padx=6)
            self.buttons[key] = b
        mk("load", "Load photo", self.load_photo)
        mk("protect", "1 · PROTECT", self.protect)
        mk("edit", "2 · DRAW", self.edit)
        mk("verify", "3 · VERIFY", self.do_verify)
        mk("reset", "Reset", self.reset)

        info = tk.Frame(self, bg=NAVY)
        info.pack(pady=(0, 6))
        self.pbc_badge = tk.Label(info, text="", font=("Segoe UI", 15, "bold"),
                                  fg=ICE, bg=NAVY, padx=14, pady=4)
        self.pbc_badge.pack(side="left", padx=6)
        mk("preview", "…or edit in Preview ↗" if platform.system() == "Darwin"
           else "…or edit in Paint ↗", self.edit_external,
           parent=info, font=("Segoe UI", 13, "bold"), pady=4)
        self.demo_btn = tk.Label(info, text="▶ Auto demo", font=("Segoe UI", 13, "bold"),
                                 bg=BLUE, fg="white", padx=18, pady=4, cursor="hand2",
                                 width=14)                 # fixed: label text changes
        self.demo_btn.bind("<Button-1>", lambda e: self.toggle_demo())
        self.demo_btn.pack(side="left", padx=6)
        self.restart_btn = tk.Label(info, text="⟲ Demo from start", font=("Segoe UI", 13, "bold"),
                                    bg=BLUE, fg="white", padx=18, pady=4, cursor="hand2")
        self.restart_btn.bind("<Button-1>", lambda e: self.start_demo(fresh=True))
        self.restart_btn.pack(side="left", padx=6)

        panes = tk.Frame(self, bg=NAVY)
        panes.pack(expand=True, fill="both", padx=16, pady=8)
        self.left_lbl = self._pane(panes, "Protected image (invisible watermark)")
        self.right_lbl = self._pane(panes, "Verification verdict")
        self.left_lbl.bind("<ButtonPress-1>",
                           lambda e: None if self.demo_on else self._draw(e, start=True))
        self.left_lbl.bind("<B1-Motion>",
                           lambda e: None if self.demo_on else self._draw(e))
        self.pen_btn = tk.Label(self.left_lbl.master, font=("Segoe UI", 13, "bold"),
                                padx=14, pady=3, cursor="hand2",
                                highlightthickness=3, highlightbackground=NAVY)
        self.pen_btn.bind("<Button-1>",
                          lambda e: None if self.demo_on else
                          (self._discard_demo(), self._set_pen(not self.pen_compliant)))
        self.pen_btn.pack(before=self.left_lbl, pady=(2, 0))
        self.right_lbl.bind("<Button-1>", self._ledger_click)
        # any click / key during the auto demo pauses it and hands control to the visitor;
        # ← / → instead step to the previous / next slide (more specific bindings win)
        self.bind_all("<Button-1>", self._demo_interrupt, add="+")
        self.bind_all("<Key>", self._demo_interrupt, add="+")
        self.bind_all("<Left>", lambda e: self._step_slide(-1))
        self.bind_all("<Right>", lambda e: self._step_slide(+1))

        self.verdict = tk.Label(self, text="", font=huge, fg="white", bg=NAVY)
        self.verdict.pack(pady=(0, 14))
        # fixed-size explainer bar above the images: demo step cards and the tile Edit
        # Ledger, with the countdown pie in its own slot — the layout never moves
        bar = tk.Frame(self, bg=PANE_BG, width=1160, height=150,
                       highlightbackground=BLUE, highlightthickness=2)
        bar.pack(before=panes, pady=(4, 6))       # read first, then look at the images
        bar.grid_propagate(False)
        bar.columnconfigure(0, weight=1)
        bar.rowconfigure(0, weight=1)
        text = tk.Frame(bar, bg=PANE_BG)
        text.grid(row=0, column=0)
        self.explain_title = tk.Label(text, font=("Segoe UI", 18, "bold"), fg="white", bg=PANE_BG)
        self.explain_title.pack()
        self.explain_body = tk.Label(text, font=("Segoe UI", 14), fg=ICE, bg=PANE_BG,
                                     wraplength=1110, justify="center")
        self.explain_body.pack(pady=(4, 0))
        # pen colour legend, shown on the slides where the two pens matter
        self.legend = tk.Frame(text, bg=PANE_BG)
        for color, txt in PEN_LEGEND:
            tk.Label(self.legend, text=txt, font=("Segoe UI", 13, "bold"), fg=color,
                     bg=PANE_BG).pack(side="left", padx=14)
        self.explain_legend = False
        # slide bullets: the active one doubles as the countdown pie; click any to jump
        self.bullets = tk.Canvas(bar, width=len(DEMO_SECTIONS) * 30 + 70, height=26,
                                 bg=PANE_BG, highlightthickness=0, cursor="hand2")
        self.bullets.grid(row=1, column=0, pady=(0, 6))
        self.bullet_arc = self.pie_job = None

        # demo cursor: a big arrow in a borderless transparent window that glides to
        # the buttons and clicks them, so viewers can follow what the auto demo does
        self.cursor_win = tk.Toplevel(self)
        self.cursor_win.overrideredirect(True)
        self.cursor_win.withdraw()
        cbg = NAVY
        try:
            if platform.system() == "Darwin":
                self.cursor_win.attributes("-transparent", True)
                cbg = "systemTransparent"
            elif platform.system() == "Windows":
                self.cursor_win.attributes("-transparentcolor", "#010203")
                cbg = "#010203"
            self.cursor_win.attributes("-topmost", True)
        except tk.TclError:
            pass
        self.cursor_win.configure(bg=cbg)
        self.cursor_imgs = [ImageTk.PhotoImage(self._cursor_image(p)) for p in (False, True)]
        self.cursor_lbl = tk.Label(self.cursor_win, image=self.cursor_imgs[0], bg=cbg,
                                   bd=0, highlightthickness=0)
        self.cursor_lbl.pack()

        self._set_buttons({"load"}, "load")
        if os.path.exists(DEFAULT_IMAGE):
            self._load(DEFAULT_IMAGE)
        self._watch()

    def _pane(self, parent, title):
        f = tk.Frame(parent, bg=NAVY)
        f.pack(side="left", expand=True, fill="both", padx=8)
        tk.Label(f, text=title, font=("Segoe UI", 13), fg=ICE, bg=NAVY).pack()
        lbl = tk.Label(f, bg=PANE_BG)
        lbl.pack(expand=True, fill="both", pady=4)
        self._clear(lbl)
        return lbl

    def _show(self, lbl, pil_img):
        if lbl is getattr(self, "right_lbl", None):
            self._cancel_compare()
        img = pil_img.copy()
        img.thumbnail((PREVIEW_W, PREVIEW_H))
        tkimg = ImageTk.PhotoImage(img)
        # fixed box (image centred) so panes never change size with the content
        lbl.configure(image=tkimg, width=PREVIEW_W, height=PREVIEW_H, text="")
        lbl.image, lbl.thumb, lbl.scale = tkimg, img, pil_img.width / img.width

    def _clear(self, lbl):
        # blank placeholder image: an image-less Label sizes width/height in
        # text units (640 chars x 460 lines), pushing the verdict off-screen
        if lbl is getattr(self, "right_lbl", None):
            self._cancel_compare()
        tkimg = ImageTk.PhotoImage(Image.new("RGB", (PREVIEW_W, PREVIEW_H), PANE_BG))
        lbl.configure(image=tkimg, width=PREVIEW_W, height=PREVIEW_H, text="")
        lbl.image = tkimg

    def _cancel_compare(self):
        if self.compare_job:
            self.after_cancel(self.compare_job)
            self.compare_job = None

    def _compare_frame(self, elapsed):
        """One frame of slide 4's measured PBC vs TrustMark comparison. Time rows grow
        in real time at the same rate for both, so the faster one really finishes first."""
        if self.compare_fonts is None:
            self.compare_fonts = [label_font(n)[0] for n in (12, 11, 10)]
        f12, f11, f10 = self.compare_fonts
        x_lab, x_bar, bar_w, x_val = 8, 112, 290, 418
        if self.compare_bg is None:       # static parts drawn once; frames add bars + values
            bg = Image.new("RGB", (PREVIEW_W, PREVIEW_H), PANE_BG)
            d = ImageDraw.Draw(bg)
            for x, color, name in ((x_bar, ICE, "PBC"), (x_bar + 70, GRAY, "TrustMark (Adobe), 1 CPU core")):
                d.rectangle([x, 9, x + 9, 18], fill=color)
                d.text((x + 14, 6), name, fill=color, font=f12)
            y = 28
            for row in RUN_COMPARE:
                d.text((x_lab, y + 7), row[0], fill="white", font=f12)
                y += 30
            y += 4
            for label, a, b in RUN_COMPARE_TEXT:
                d.text((x_lab, y), label, fill="white", font=f12)
                d.text((x_bar, y), a, fill=ICE, font=f12)
                d.text((x_bar + 170, y), b, fill=GRAY, font=f12)
                y += 20
            for line in ("Measured on this laptop (i7-4870HQ, CPU only), 978×678 photo, encode()/verify() calls only.",
                         "PBC: NumPy, 1 core. TrustMark Q, PyTorch 2.2, 1 core (4 cores: 294 ms / 116 ms)."):
                d.text((x_lab, y + 6), line, fill=GRAY, font=f10)
                y += 14
            self.compare_bg = bg
        im = self.compare_bg.copy()
        d = ImageDraw.Draw(im)
        k = min(1.0, elapsed / 1.2)
        ease = k * k * (3 - 2 * k)
        y = 28
        for label, unit, pbc, tm, lower, real in RUN_COMPARE:
            top = max(pbc, tm)
            row_done = elapsed >= max(pbc, tm) if real else k >= 1.0
            for i, (v, other, color) in enumerate(((pbc, tm, ICE), (tm, pbc, GRAY))):
                shown = min(v, elapsed) if real else v * ease
                by = y + 3 + i * 12
                if shown > 0:
                    d.rectangle([x_bar, by, x_bar + bar_w * shown / top, by + 8], fill=color)
                if unit == "s":
                    txt = f"{shown * 1000:.0f} ms" if v < 1 else f"{shown:.2f} s"
                elif unit == "MB":
                    txt = "0 MB · no model" if row_done and v == 0 else f"{shown:.0f} MB"
                else:
                    txt = f"{shown:.1f} dB"
                d.text((x_val, by - 3), txt, font=f11, fill=color)
                if row_done and (v <= other if lower else v >= other):   # ◀ marks the better value
                    d.polygon([(x_val - 4, by - 1), (x_val - 4, by + 8), (x_val - 11, by + 3.5)],
                              fill="white")
            y += 30
        return im

    def _run_compare(self, t0=None):
        """Slide 4: animate the measured comparison in the right pane."""
        if t0 is None:
            self._show(self.right_lbl, self._compare_frame(0.0))
            t0 = time.perf_counter()
        elapsed = time.perf_counter() - t0
        self.right_lbl.image.paste(self._compare_frame(elapsed))
        if elapsed < max(max(r[2], r[3]) for r in RUN_COMPARE if r[5]) + 0.1:
            self.compare_job = self.after(50, self._run_compare, t0)
        else:
            self.compare_job = None

    def _pane_text(self, lbl, text):
        """Text card on an empty pane (used for the resource readout)."""
        self._clear(lbl)
        lbl.configure(text=text, compound="center", fg=ICE, font=("Segoe UI", 14),
                      justify="center", wraplength=PREVIEW_W - 24)   # never widen the pane

    def _show_grid(self, res, hold=6000):
        """Flash the tile grid over the protected image, each tile labelled with its
        size and the number of blocks in its chain (e.g. 124×138 → 198), then fade back."""
        self._cancel_grid()
        img = self.edit_img.copy()
        W, H = img.size
        tw, th = W // res.cols, H // res.rows          # same geometry as the decoder
        s = max(1.0, W / PREVIEW_W, H / PREVIEW_H)      # image px per on-screen px
        font, ascii_only = label_font(round(12 * s))
        times, arrow = ("x", "->") if ascii_only else ("×", "→")
        d = ImageDraw.Draw(img)
        for ty in range(res.rows):
            for tx in range(res.cols):
                x0, y0 = tx * tw, ty * th
                x1 = W if tx == res.cols - 1 else x0 + tw   # edge tiles take the remainder
                y1 = H if ty == res.rows - 1 else y0 + th
                d.rectangle([x0, y0, x1 - 1, y1 - 1], outline="white", width=max(2, round(1.5 * s)))
                cx, cy, gap = (x0 + x1) / 2, (y0 + y1) / 2, 7 * s
                for text, dy in ((f"{x1 - x0}{times}{y1 - y0}", -gap),
                                 (f"{arrow} {res.tile_results[ty][tx].block_count}", gap)):
                    d.text((cx, cy + dy), text, fill="white", font=font, anchor="mm",
                           stroke_width=max(2, round(1.5 * s)), stroke_fill=NAVY)
        self._show(self.left_lbl, img)
        self.grid_job = self.after(hold, self._fade_grid, 1)

    def _fade_grid(self, step):
        lbl = self.left_lbl
        if step == 1:
            clean = self.edit_img.copy()
            clean.thumbnail((PREVIEW_W, PREVIEW_H))
            self.grid_from, self.grid_to = lbl.thumb.copy(), clean
        if step <= 5:
            lbl.image.paste(Image.blend(self.grid_from, self.grid_to, step / 5))
            self.grid_job = self.after(70, self._fade_grid, step + 1)
        else:
            self._hide_grid()

    def _cancel_grid(self):
        if self.grid_job:
            self.after_cancel(self.grid_job)
            self.grid_job = None
            return True
        return False

    def _hide_grid(self):
        """Drop the grid now (e.g. DRAW pressed) so strokes land on the clean image."""
        if self._cancel_grid():
            self._show(self.left_lbl, self.edit_img)

    def _set_buttons(self, enabled, nxt):
        """Dim steps that don't apply yet; highlight the next step in white."""
        for key, b in self.buttons.items():
            b.enabled = key in enabled
            if key == nxt:
                bg, fg = "white", NAVY
            elif b.enabled:
                bg, fg = BLUE, "white"
            else:
                bg, fg = BTN_OFF_BG, BTN_OFF_FG
            b.configure(bg=bg, fg=fg, cursor="hand2" if b.enabled else "arrow")

    def _show_detection(self, img, source, result=None):
        """Badge: is a PBC chain present at all (any tile not ABSENT)?"""
        self.last_result = result if result is not None else verify(img)
        tiles = [t for row in self.last_result.tile_results for t in row]
        found = sum(t.status != TileStatus.ABSENT for t in tiles)
        if found:
            self.pbc_badge.configure(
                text=f"●  PBC chain detected in {found}/{len(tiles)} tiles  ({source})",
                bg=GREEN, fg="white")
        else:
            self.pbc_badge.configure(
                text=f"○  No PBC chain detected  ({source})", bg=BTN_OFF_BG, fg=ICE)
        return found

    def _watch(self):
        """Poll every second for an external editor's save — in place or as
        Preview's '… copy.png' — and load it back so VERIFY reads that file."""
        if self.work_mtime is not None and not self.busy:   # PROTECT's own save isn't an edit
            stem = os.path.splitext(os.path.basename(WORK_PNG))[0]
            try:
                files = [f for d in WATCH_DIRS
                         for f in glob.glob(os.path.join(glob.escape(d), stem + "*.png"))
                         if os.path.getmtime(f) > self.work_mtime]
                path = max(files, key=os.path.getmtime) if files else None
                pil = Image.open(path).convert("RGB") if path else None
            except Exception:            # editor still writing — retry next tick
                path = pil = None
            if pil is not None:
                self.work_mtime = os.path.getmtime(path)
                self.verify_path, self.edit_img, self.dirty = path, pil, False
                self._discard_demo()
                self._cancel_grid()
                self._show(self.left_lbl, pil)
                self._show_detection(np.array(pil, dtype=np.uint8),
                                     f"{os.path.basename(path)}, just saved")
                self.status.configure(
                    text="Edit saved and loaded back — press 3 · VERIFY to see where.")
                self._set_buttons(ALL_STEPS, "verify")
        self.after(1000, self._watch)

    def _draw(self, e, start=False):
        """In-window pen: paint on the preview and on the full-size edit image."""
        if not self.drawing or self.edit_img is None or self.busy:
            return
        if not self.demo_on:             # a visitor is drawing
            self._discard_demo()
        lbl = self.left_lbl
        x = e.x - (lbl.winfo_width() - lbl.image.width()) / 2    # image is centred
        y = e.y - (lbl.winfo_height() - lbl.image.height()) / 2
        p0 = (x, y) if start or self.last_pt is None else self.last_pt
        color = BRUSH_EDITOR if self.pen_compliant else BRUSH
        targets = [(lbl.thumb, 1, color), (self.edit_img, lbl.scale, color)]
        if self.pen_compliant:           # remember where the PBC-aware editor touched
            if self.mask_owner is not self.edit_img:
                self.mask_img = Image.new("L", self.edit_img.size, 0)
                self.mask_owner = self.edit_img
            targets.append((self.mask_img, lbl.scale, 255))
        for im, s, fill in targets:
            w = max(1, round(BRUSH_PX * s))
            d = ImageDraw.Draw(im)
            d.line([(p0[0] * s, p0[1] * s), (x * s, y * s)], fill=fill, width=w)
            d.ellipse([x * s - w / 2, y * s - w / 2, x * s + w / 2, y * s + w / 2],
                      fill=fill)
        lbl.image.paste(lbl.thumb)
        self.last_pt, self.dirty = (x, y), True

    def _set_pen(self, compliant):
        self.pen_compliant = compliant
        if compliant:
            self.pen_btn.configure(text="Pen: PBC-aware editor → YELLOW · click to switch",
                                   bg=BRUSH_EDITOR, fg=NAVY)
        else:
            self.pen_btn.configure(text="Pen: raw tamper → RED · click to switch",
                                   bg=BRUSH, fg="white")

    def _explain(self, title, body, mono=False, legend=False):
        """Fill the explainer bar (fixed size, so nothing else moves)."""
        self.explain_title.configure(text=title)
        self.explain_body.configure(text=body, font=("Menlo", 13) if mono else ("Segoe UI", 14),
                                    justify="left" if mono else "center")
        self.explain_mono, self.explain_legend = mono, legend
        if legend:
            self.legend.pack(pady=(8, 0))
        else:
            self.legend.pack_forget()

    def _bar_read_ms(self):
        """How long the bar's current text stays up in the auto demo: absorb time +
        reading time (ledger text, with hex IDs and block ranges, reads slower)."""
        shown = [self.explain_title.cget("text"), self.explain_body.cget("text")]
        if self.explain_legend:
            shown += [txt for _, txt in PEN_LEGEND]
        words = len(" ".join(shown).split())
        ms = ABSORB_MS + words * 60000 / READING_WPM * (1.3 if self.explain_mono else 1.0)
        return int(min(33000, max(8300, ms)) * self.pace)

    def _explain_default(self):
        self._explain("Try it:   1 · PROTECT   →   2 · DRAW   →   3 · VERIFY",
                      "Switch pens with the button above the left image. After VERIFY, "
                      "click any tile to read its Edit Ledger.", legend=True)

    def _ledger_candidates(self):
        """(logged edits, raw tampers, intact) tiles of the last verification.
        A YELLOW tile only counts as a logged edit if its ledger names an
        originator other than the image's original encoder."""
        tiles = [t for row in self.verify_result.tile_results for t in row]
        logged = [t for t in tiles if t.status == TileStatus.YELLOW and any(
            e.originator_id != self.base_oid for e in extract_edit_ledger(t))]
        red = [t for t in tiles if t.status == TileStatus.RED]
        green = [t for t in tiles if t.status == TileStatus.GREEN]
        return logged, red, green

    def _show_ledger(self, tx, ty):
        """Show one tile's Edit Ledger in the explainer bar and outline the tile."""
        if self._ff():                   # jumping: don't flash earlier slides' ledgers
            return
        res = self.verify_result
        t = res.tile_results[ty][tx]
        title = f"EDIT LEDGER · tile ({tx},{ty}) · {t.status.name}"
        if t.status == TileStatus.RED:      # corrupted blocks would read as fake entries
            lines = ["Raw pixel change: chain broken, so no trustworthy ledger."]
        elif t.status == TileStatus.ABSENT:
            lines = ["No PBC data in this tile."]
        else:
            entries = extract_edit_ledger(t)
            if t.status == TileStatus.GREEN:
                lines = ["Intact chain:"]
            elif any(e.originator_id != self.base_oid for e in entries):
                lines = ["Re-encoded by a PBC-aware editor, which recorded itself:"]
            elif (t.blocks and t.blocks[0].status == BlockStatus.YELLOW
                  and all(b.status == BlockStatus.GREEN for b in t.blocks[1:])):
                # intact chain whose genesis doesn't match this tile position
                lines = ["Intact chain, but its first block doesn't match this tile's position:",
                         "copied, swapped or re-encoded here, and no editor recorded it."]
            else:                           # e.g. a raw stroke that wiped whole blocks
                lines = ["Chain broken, and no editor recorded itself in the ledger:"]
            for e in entries:
                who = KNOWN_IDS.get(e.originator_id, "unregistered")
                lines.append(f"blocks {e.start_block}–{e.end_block}  {e.opcode_name:<12}  "
                             f"oid 0x{e.originator_id:08X} \"{who}\"")
            lines.append("oid = first 32 bits of SHA-256(name) · self-asserted, not a signature")
        self._explain(title, "\n".join(lines), mono=True)
        tw, th = res.width // res.cols, res.height // res.rows   # decoder geometry
        x1 = res.width if tx == res.cols - 1 else (tx + 1) * tw
        y1 = res.height if ty == res.rows - 1 else (ty + 1) * th
        s = self.right_lbl.scale
        im = self.verdict_thumb.copy()
        ImageDraw.Draw(im).rectangle([tx * tw / s, ty * th / s, x1 / s - 1, y1 / s - 1],
                                     outline="white", width=3)
        self.right_lbl.image.paste(im)

    def _ledger_click(self, e):
        if self.demo_on or self.busy or self.verify_result is None:
            return
        lbl, res = self.right_lbl, self.verify_result
        x = (e.x - (lbl.winfo_width() - lbl.image.width()) / 2) * lbl.scale
        y = (e.y - (lbl.winfo_height() - lbl.image.height()) / 2) * lbl.scale
        if 0 <= x < res.width and 0 <= y < res.height:
            self._show_ledger(min(int(x // (res.width // res.cols)), res.cols - 1),
                              min(int(y // (res.height // res.rows)), res.rows - 1))

    def _save_edits(self):
        """Write the in-window drawing to WORK_PNG; VERIFY reads it back from disk.
        Strokes from the PBC-aware pen are re-encoded first (encode_region), as a
        compliant editor would do on save."""
        if self.mask_owner is self.edit_img and self.mask_img.getbbox():
            arr = encode_region(np.array(self.edit_img, dtype=np.uint8),
                                np.array(self.mask_img) > 0, EDITOR_ID, OpCode.EDIT_RETOUCH)
            self.edit_img = Image.fromarray(arr)
        self.edit_img.save(WORK_PNG, compress_level=1)    # lossless, faster
        self.work_mtime = os.path.getmtime(WORK_PNG)
        self.verify_path, self.dirty = WORK_PNG, False

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE))
        self.original = np.array(img, dtype=np.uint8)
        self.protected = None
        self.work_mtime, self.dirty, self.drawing = None, False, False
        self.left_lbl.configure(cursor="")
        self._set_pen(False)
        self.copy_target = None
        self._cancel_grid()
        self._show(self.left_lbl, img)
        self._clear(self.right_lbl)
        self.verify_result = None
        if not self.demo_on:
            self._explain_default()
        self.verdict.configure(text="")
        if self._show_detection(self.original, "loaded photo"):
            # already carries a chain (e.g. the edited copy): not a reset, verify it
            self.verify_path, self.edit_img = path, img
            self.status.configure(text=f"Loaded {os.path.basename(path)} — it carries "
                                       f"a PBC chain. Press 3 · VERIFY.")
            self._set_buttons(ALL_STEPS, "verify")
        else:
            self.verify_path, self.edit_img = None, None
            self.status.configure(text=f"Loaded {os.path.basename(path)} "
                                       f"({img.width}x{img.height}). Press 1 · PROTECT.")
            self._set_buttons({"load", "protect", "reset"}, "protect")

    def load_photo(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if path:
            self._load(path)

    def protect(self):
        if self.original is None:
            return messagebox.showinfo("PBC", "Load a photo first.")
        if self.busy:
            return
        self.busy_t0 = time.perf_counter()   # click: the card reports click → result on screen
        self._hide_grid()
        self.status.configure(text="Encoding PBC chains into the pixels…")
        original = self.original

        def work():                      # worker thread: no Tk calls in here
            wall0, cpu0 = time.perf_counter(), time.thread_time()
            protected = encode(original, originator=ORIGINATOR)
            enc_s, enc_cpu = time.perf_counter() - wall0, time.thread_time() - cpu0
            t = time.perf_counter()
            Image.fromarray(protected).save(WORK_PNG, compress_level=1)   # lossless, faster
            save_s, t = time.perf_counter() - t, time.perf_counter()
            check = verify(np.array(Image.open(WORK_PNG).convert("RGB"), dtype=np.uint8))
            return protected, enc_s, enc_cpu, check, save_s, time.perf_counter() - t
        self._run_busy("protect", "Writing hash-linked chains into every tile",
                       work, self._protect_done)

    def _protect_done(self, protected, enc_s, enc_cpu, check, save_s, check_s):
        self.protected, self.last_encode_s = protected, enc_s
        mse = float(np.mean((self.original.astype(float) -
                             self.protected.astype(float)) ** 2))
        psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float("inf")
        self._show(self.left_lbl, Image.fromarray(self.protected))
        self._clear(self.right_lbl)
        self.verify_result = None
        if not self.demo_on:
            self._explain_default()
        self.verdict.configure(text="")
        self.work_mtime = os.path.getmtime(WORK_PNG)
        self.verify_path, self.dirty, self.drawing = WORK_PNG, False, False
        self.edit_img = Image.fromarray(self.protected)
        self.left_lbl.configure(cursor="")
        self._show_detection(None, "saved PNG", result=check)
        self.status.configure(
            text=f"Protected. PSNR {psnr:.1f} dB — can you see the watermark? "
                 f"Now press 2 · DRAW and scribble anything.")
        self._set_buttons(ALL_STEPS, "edit")

        # resource readout: measured on this machine, comparison worded as in the paper
        res = self.last_result
        blocks = sum(t.block_count for row in res.tile_results for t in row)
        h, w = self.protected.shape[:2]
        model, cpu = self.machine
        ram = peak_ram_mb()
        ram_txt = f"\npeak RAM {ram:.0f} MB (whole app)" if ram else ""
        self._show_grid(res)             # draw the result first so the timing includes it
        self.update_idletasks()
        # what the audience watched: click → result on screen = worker steps + UI ("display")
        watched_s = time.perf_counter() - self.busy_t0
        display_ms = max(0.0, watched_s - enc_s - save_s - check_s) * 1000
        self._pane_text(self.right_lbl,
                        f"⚡  Protected in {watched_s * 1000:.0f} ms (the step you just watched)\n"
                        f"embed {enc_s * 1000:.0f} · save PNG {save_s * 1000:.0f} · re-read + check "
                        f"{check_s * 1000:.0f} · display {display_ms:.0f} ms\n\n"
                        f"{w}×{h} px · {w * h / 1e6:.2f} MP · PSNR {psnr:.1f} dB\n"
                        f"{res.cols}×{res.rows} = {res.cols * res.rows} tiles · {blocks:,} blocks\n"
                        f"embed CPU {enc_cpu * 1000:.0f} ms, single-threaded · no GPU{ram_txt}\n"
                        f"{model} · {cpu}\n\n"
                        "Neural watermarks run trained networks on GPUs\n"
                        "(EditGuard: 5.45 M parameters, RTX 3090 Ti,\n"
                        "as reported by its authors).\n"
                        "PBC is learning-free: NumPy + Pillow only.")

    def edit(self):
        if self.edit_img is None:
            return messagebox.showinfo("PBC", "Protect the image first (step 1).")
        self._hide_grid()
        self.drawing = True
        self.left_lbl.configure(cursor="pencil")
        self.status.configure(
            text="Draw on the LEFT image (switch the pen above it for a logged edit), "
                 "then press 3 · VERIFY.")
        self._set_buttons(ALL_STEPS, "verify")

    def edit_external(self):
        if self.edit_img is None:
            return messagebox.showinfo("PBC", "Protect the image first (step 1).")
        self._hide_grid()
        self._save_edits()               # editor opens the current image
        open_in_editor(WORK_PNG)
        self.status.configure(
            text="Draw / erase / clone anything in the editor, then SAVE "
                 "(keep PNG!) — it loads back here automatically.")
        self._set_buttons(ALL_STEPS, "verify")

    def do_verify(self):
        if self.busy:
            return
        self.busy_t0 = time.perf_counter()   # click: the verdict reports click → result on screen
        if self.dirty:
            self._save_edits()           # save the drawing; verify reads it from disk
        if self.verify_path is None or not os.path.exists(self.verify_path):
            return messagebox.showinfo("PBC", "Nothing to verify yet.")
        self._hide_grid()
        self.status.configure(text="Scanning LSB chains, verifying every tile…")
        path = self.verify_path

        def work():                      # worker thread: reads the file back from disk
            img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
            t = time.perf_counter()
            result = verify(img)
            check_s = time.perf_counter() - t
            return result, generate_overlay(img, result, opacity=0.45), check_s
        self._run_busy("verify", "Re-reading every chain from the saved file",
                       work, self._verify_done)

    def _run_busy(self, key, label, work, done):
        """Run work() in a worker thread while a tile-by-tile sweep animates over the
        left image and the status line ticks; then call done(*result) on the Tk thread."""
        box = {}

        def target():
            try:
                box["out"] = work()
            except Exception as ex:      # reported on the Tk thread below
                box["err"] = ex
        worker = threading.Thread(target=target, daemon=True)
        lbl = self.left_lbl
        base, s = lbl.thumb.convert("RGBA"), lbl.scale
        cols, rows, tw, th = compute_grid(round(base.width * s), round(base.height * s), 128)
        ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw, shown = ImageDraw.Draw(ov), [-1]

        def rect(j):
            x, y = (j % cols) * tw / s, (j // cols) * th / s
            return [x, y, x + tw / s - 1, y + th / s - 1]
        self.busy, t0 = True, time.perf_counter()
        worker.start()

        def frame():
            elapsed = time.perf_counter() - t0
            if worker.is_alive():
                i = min(cols * rows - 1, int(elapsed / self.est_s[key] * cols * rows))
                if i != shown[0]:        # repaint only when the sweep reaches a new tile
                    for j in range(max(shown[0], 0), i):     # finished: thin outline
                        draw.rectangle(rect(j), fill=(0, 0, 0, 0),
                                       outline=(228, 236, 244, 140), width=1)
                    draw.rectangle(rect(i), fill=(0, 194, 255, 90),          # current: lit
                                   outline=(255, 255, 255, 255), width=2)
                    shown[0] = i
                    lbl.image.paste(Image.alpha_composite(base, ov).convert("RGB"))
                spin = "◐◓◑◒"[int(elapsed * 8) % 4]
                self.status.configure(text=f"{spin}  {label}…  {elapsed:.1f} s")
                self.after(20, frame)
                return
            lbl.image.paste(lbl.thumb)
            self.busy = False
            self.est_s[key] = self.last_busy_s = elapsed   # next sweep matches this pace
            if "err" in box:
                return messagebox.showerror("PBC", f"{key} failed: {box['err']}")
            done(*box["out"])
        frame()

    def _verify_done(self, result, overlay, check_s):
        self._show(self.right_lbl, overlay)
        self.verify_result, self.verdict_thumb = result, self.right_lbl.thumb.copy()
        # the image's original encoder = most common originator among intact tiles
        greens = [t.originator_id for row in result.tile_results for t in row
                  if t.status == TileStatus.GREEN]
        self.base_oid = max(set(greens), key=greens.count) if greens else None

        counts = {s: 0 for s in TileStatus}
        for row in result.tile_results:
            for t in row:
                counts[t.status] += 1
        g, y, r, a = (counts[TileStatus.GREEN], counts[TileStatus.YELLOW],
                      counts[TileStatus.RED], counts[TileStatus.ABSENT])
        if r == 0 and a == 0 and y == 0:
            txt, col = f"INTACT — {g} tiles GREEN", GREEN
        elif r > 0 or a > 0:
            txt, col = (f"TAMPERED — {r + a} tile(s) flagged, "
                        + (f"{y} YELLOW, " if y else "")
                        + f"{g} still GREEN"), RED
        else:
            unlogged = y - len(self._ledger_candidates()[0])
            if unlogged:                 # e.g. a copied or swapped tile: nobody logged it
                txt, col = (f"MOVED OR RE-ENCODED WITHOUT A LEDGER ENTRY — "
                            f"{unlogged} of {y} YELLOW tile(s)"), YELLOW
            else:
                txt, col = f"EDITED & LOGGED — {y} tiles YELLOW", YELLOW
        self.verdict.configure(text=txt, fg=col)
        self.status.configure(
            text="Only the tiles you touched change — by design, tiles never affect each "
                 "other. Click a tile for its Edit Ledger, or Reset for the next visitor.")
        self._set_buttons(ALL_STEPS, "reset")
        # open the most telling tile's ledger: a logged edit, else a tamper, else intact
        logged, red, green = self._ledger_candidates()
        pick = (logged + red + green or [None])[0]
        self.ledger_pick = (pick.tx, pick.ty) if pick else None
        if pick and not self.demo_on:    # the demo shows it on the Verdict slide instead
            self._show_ledger(pick.tx, pick.ty)
        # timing = the step the audience watched (click → everything on screen), plus verify()
        self.update_idletasks()
        watched_s = time.perf_counter() - self.busy_t0
        self.verdict.configure(text=f"{txt}  ·  {watched_s * 1000:.0f} ms "
                                    f"(verify {check_s * 1000:.0f} ms)")

    def reset(self):
        if os.path.exists(DEFAULT_IMAGE):
            self._load(DEFAULT_IMAGE)

    # ---- auto demo: the visitor flow on a loop, for the TV between visitors ----

    def toggle_demo(self):
        self.stop_demo() if self.demo_on else self.start_demo()

    def start_demo(self, fresh=False):
        """Start the auto demo, or resume a paused one at the step it was on."""
        if self.busy or not os.path.exists(DEFAULT_IMAGE):
            return
        if self.demo_on:
            if not fresh:
                return
            self.stop_demo()
        resume = self.demo_gen is not None and not fresh
        if not resume:
            self.demo_gen, self.demo_delay, self.demo_bar = self._demo_script(), 0, None
            self.demo_section = self.demo_ff = 0
            self.cursor_pos = None       # first glide starts from the window centre
        self.demo_on = True
        self.demo_btn.configure(text="■ Pause demo", bg="white", fg=NAVY)
        self._draw_bullets()
        if not resume:
            return self._demo_tick()
        if self.demo_bar:                # the card that was up, with a full reading time again
            self._explain(*self.demo_bar)
        if self.cursor_pos:
            self._cursor_to(*self.cursor_pos)
            self.cursor_win.deiconify()
            self.cursor_win.lift()
        self.status.configure(text="Auto demo — click anywhere to pause.")
        self._pie(self.demo_delay)
        self.demo_job = self.after(self.demo_delay, self._demo_tick)

    def stop_demo(self):
        """Pause where it is — everything stays on screen, Resume continues this step."""
        if not self.demo_on:
            return
        self.demo_on, self.pie_span = False, False
        if self.demo_job:
            self.after_cancel(self.demo_job)
            self.demo_job = None
        self.demo_bar = (self.explain_title.cget("text"), self.explain_body.cget("text"),
                         self.explain_mono, self.explain_legend)
        self.demo_btn.configure(text="▶ Resume demo", bg=BLUE, fg="white")
        self._draw_bullets()             # stay visible: a paused demo can jump too
        self._pie(0)
        self.cursor_win.withdraw()
        self.status.configure(text="Demo paused — ▶ Resume continues here, ⟲ restarts. "
                                   "Or take over: follow the white button.")

    def _discard_demo(self):
        """A visitor changed the app's state, so a paused demo can't safely resume."""
        if not self.demo_on and self.demo_gen is not None:
            self.demo_gen, self.demo_bar, self.demo_ff = None, None, 0
            self.demo_btn.configure(text="▶ Auto demo")
            self._draw_bullets()

    def _ff(self):
        """True while a running jump replays the slides before its target."""
        return self.demo_on and self.demo_section < self.demo_ff

    def jump_demo(self, section):
        """Jump the auto demo to a slide. Its state is rebuilt first: the round is
        replayed from the start without pauses, so reset / PROTECT / the strokes /
        VERIFY really run and the slide shows genuine results."""
        if self.busy:                    # PROTECT / VERIFY still finishing
            return
        if self.demo_on:
            self.stop_demo()
        self.demo_gen, self.demo_delay, self.demo_bar = self._demo_script(), 0, None
        self.demo_section, self.demo_ff, self.cursor_pos = 0, section, None
        if section:
            self._explain(f"Jumping to slide {section + 1} · {DEMO_SECTIONS[section]}",
                          "Replaying the earlier steps for real, so this slide shows "
                          "genuine results…")
        self.start_demo()                # resumes the fresh generator right away

    def _step_slide(self, delta):
        """← / →: jump to the previous / next auto-demo slide (wrapping around)."""
        if not self.demo_on and self.demo_gen is None:
            return
        cur = max(self.demo_section, self.demo_ff)
        self.jump_demo((cur + delta) % len(DEMO_SECTIONS))

    def _draw_bullets(self):
        """Slide bullets (shown while a demo runs or is paused): the active one is a
        ring whose inner pie counts down the slide; click any bullet to jump there."""
        c = self.bullets
        c.delete("all")
        self.bullet_arc = None
        if not self.demo_on and self.demo_gen is None:
            return
        cur, n = max(self.demo_section, self.demo_ff), len(DEMO_SECTIONS)  # a jump shows its target
        for i in range(n):
            x, y, tag = 15 + i * 30, 13, f"slide{i}"
            c.create_rectangle(x - 15, 0, x + 15, 26, fill=PANE_BG, outline="", tags=tag)  # hit area
            if i == cur:
                c.create_oval(x - 10, y - 10, x + 10, y + 10, outline="white", width=2, tags=tag)
                self.bullet_arc = c.create_arc(x - 7, y - 7, x + 7, y + 7, start=90, extent=359.9,
                                               fill="white", outline="", tags=tag)
            else:
                c.create_oval(x - 5, y - 5, x + 5, y + 5, outline=ICE, width=1.5, tags=tag,
                              fill=BTN_OFF_FG if i < cur else PANE_BG)
            c.tag_bind(tag, "<Button-1>", lambda e, i=i: self.jump_demo(i))
        c.create_text(15 + n * 30 + 8, 13, text=f"{cur + 1} / {n}", anchor="w", fill=ICE,
                      font=("Segoe UI", 13, "bold"))

    def _demo_interrupt(self, e):
        if self.demo_on and e.widget not in (self.demo_btn, self.restart_btn, self.bullets):
            self.stop_demo()

    def _demo_tick(self):
        if not self.demo_on:
            return
        while True:
            try:
                step = next(self.demo_gen)
            except StopIteration:        # one round done: start the next
                self.demo_gen, self.demo_ff = self._demo_script(), 0
                continue
            if isinstance(step, tuple):  # ("section", i): a new slide starts
                self.demo_section, self.pie_span = step[1], False
                self._draw_bullets()
                continue
            break
        delay = step
        if self._ff():                   # replaying towards a jump target: no pauses
            delay = 40 if self.busy else 0
        self.demo_delay = delay
        if not self.pie_span:            # a slide-wide countdown keeps running on its own
            self._pie(delay)
        self.demo_job = self.after(delay, self._demo_tick)

    def _pie(self, total_ms, t0=None):
        """Count down pauses of 1.5 s or more on the active bullet's pie; for short
        pauses (clicks, strokes) or when paused, the pie stays full."""
        if self.pie_job:
            self.after_cancel(self.pie_job)
            self.pie_job = None
        if self.bullet_arc is None:
            return
        t0 = time.perf_counter() if t0 is None else t0
        left = total_ms / 1000 - (time.perf_counter() - t0)
        if self.pie_span and self.demo_on and left <= 0:   # time's up, actions still finishing
            self.bullets.itemconfigure(self.bullet_arc, extent=0.1)
            return
        if total_ms < 1500 or left <= 0 or not self.demo_on:
            self.bullets.itemconfigure(self.bullet_arc, extent=359.9)
            return
        self.bullets.itemconfigure(self.bullet_arc,
                                   extent=max(0.1, 359.9 * left * 1000 / total_ms))
        self.pie_job = self.after(33, self._pie, total_ms, t0)

    @staticmethod
    def _cursor_image(pressed):
        """Big white arrow with a navy outline; the pressed frame adds a click ripple."""
        im = Image.new("RGBA", (110, 120), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        h = CURSOR_HOT
        if pressed:
            d.ellipse([h - 24, h - 24, h + 24, h + 24], outline=BRUSH_EDITOR, width=5)
        k = 1.6 * (0.88 if pressed else 1.0)
        arrow = [(0, 0), (0, 40), (10, 31), (17, 46), (24, 43), (17, 28), (30, 28)]
        d.polygon([(h + x * k, h + y * k) for x, y in arrow], fill="white", outline=NAVY, width=3)
        return im

    def _cursor_to(self, x, y, pressed=False):
        self.cursor_pos = (x, y)
        self.cursor_lbl.configure(image=self.cursor_imgs[pressed])
        self.cursor_win.geometry(f"+{int(x) - CURSOR_HOT}+{int(y) - CURSOR_HOT}")

    def _demo_point(self, x, y, glide_ms=900):
        """Glide the demo cursor to screen point (x, y). Generator: yields ms."""
        if self._ff():
            self.cursor_pos = (x, y)
            return
        if self.cursor_pos is None:
            self.cursor_pos = (self.winfo_rootx() + self.winfo_width() // 2,
                               self.winfo_rooty() + self.winfo_height() // 2)
        if not self.cursor_win.winfo_viewable():
            self._cursor_to(*self.cursor_pos)
            self.cursor_win.deiconify()
            self.cursor_win.lift()
        x0, y0 = self.cursor_pos
        t_start = time.perf_counter()     # time-based: late timer ticks don't slow the glide
        while True:
            k = min(1.0, (time.perf_counter() - t_start) * 1000 / glide_ms)
            e = k * k * (3 - 2 * k)       # ease in-out
            self._cursor_to(x0 + (x - x0) * e, y0 + (y - y0) * e)
            if k >= 1.0:
                return
            yield 16

    def _demo_click(self, widget=None, point=None):
        """Glide onto a widget's centre (or a screen point) and click it. Generator."""
        if widget is not None:
            point = (widget.winfo_rootx() + widget.winfo_width() // 2,
                     widget.winfo_rooty() + widget.winfo_height() // 2)
        if self._ff():                   # jumping: no glide, no click animation
            self.cursor_pos = point
            return
        yield from self._demo_point(*point)
        if widget is not None:
            self._flash(widget)
        self._cursor_to(*point, pressed=True)
        yield 180
        self._cursor_to(*point)
        yield 500

    def _demo_copy_tile(self, strokes):
        """Copy-paste attack (generator): drag an exact copy of one untouched tile onto
        another untouched tile of the same size, then drop its real pixels there."""
        lbl = self.left_lbl
        s, (W, H) = lbl.scale, self.edit_img.size
        cols, rows, tw, th = compute_grid(W, H, 128)   # the decoder's tile geometry

        def untouched(tx, ty):            # no stroke box overlaps this tile (preview space)
            x0, y0, x1, y1 = tx * tw / s, ty * th / s, (tx + 1) * tw / s, (ty + 1) * th / s
            return all(x1 < a or x0 > c or y1 < b or y0 > d for a, b, c, d in strokes)
        # interior tiles only: the last column / row are a few pixels larger
        tiles = [(tx, ty) for ty in range(rows - 1) for tx in range(cols - 1) if untouched(tx, ty)]
        if len(tiles) < 2:
            return
        src = random.choice(tiles)
        dst = random.choice([t for t in tiles if abs(t[0] - src[0]) + abs(t[1] - src[1]) >= 3]
                            or [t for t in tiles if t != src])
        crop = self.edit_img.crop((src[0] * tw, src[1] * th, (src[0] + 1) * tw, (src[1] + 1) * th))
        gw, gh = round(tw / s), round(th / s)
        if not self._ff():
            ox = (lbl.winfo_width() - lbl.image.width()) / 2
            oy = (lbl.winfo_height() - lbl.image.height()) / 2
            rx, ry = lbl.winfo_rootx() + ox, lbl.winfo_rooty() + oy
            (sx, sy), (dx, dy) = [((t[0] + .5) * tw / s, (t[1] + .5) * th / s) for t in (src, dst)]
            yield from self._demo_point(rx + sx, ry + sy)
            self._cursor_to(rx + sx, ry + sy, pressed=True)
            yield 300
            base, ghost = lbl.thumb.copy(), crop.resize((gw, gh))
            t_start = time.perf_counter()
            while True:                   # drag a copy of the tile, outlined, ~1.2 s
                k = min(1.0, (time.perf_counter() - t_start) / 1.2)
                e = k * k * (3 - 2 * k)
                cx, cy = sx + (dx - sx) * e, sy + (dy - sy) * e
                frame, at = base.copy(), (round(cx - gw / 2), round(cy - gh / 2))
                frame.paste(ghost, at)
                ImageDraw.Draw(frame).rectangle([at[0], at[1], at[0] + gw - 1, at[1] + gh - 1],
                                                outline="white", width=2)
                lbl.image.paste(frame)
                self._cursor_to(rx + cx, ry + cy, pressed=True)
                if k >= 1.0:
                    break
                yield 30
        # drop: the exact pixels, LSBs and all, go into the image that VERIFY will save
        self.edit_img.paste(crop, (dst[0] * tw, dst[1] * th))
        self.copy_target, self.dirty = dst, True
        self._show(lbl, self.edit_img)
        x0, y0 = dst[0] * tw / s, dst[1] * th / s        # white box: on screen only, not saved
        ImageDraw.Draw(lbl.thumb).rectangle([x0, y0, x0 + gw - 1, y0 + gh - 1],
                                            outline="white", width=2)
        lbl.image.paste(lbl.thumb)
        if not self._ff():
            self._cursor_to(rx + dx, ry + dy)
            yield 500

    def _tile_point(self, tx, ty):
        """Screen coordinates of a tile's centre on the verdict pane."""
        res, lbl = self.verify_result, self.right_lbl
        s, tw, th = lbl.scale, res.width // res.cols, res.height // res.rows
        return (lbl.winfo_rootx() + (lbl.winfo_width() - lbl.image.width()) / 2 + (tx + .5) * tw / s,
                lbl.winfo_rooty() + (lbl.winfo_height() - lbl.image.height()) / 2 + (ty + .5) * th / s)

    def _flash(self, widget, n=8):
        """Pulse a thin border on the button the auto demo is about to 'press'."""
        if n <= 0 or not self.demo_on:
            widget.configure(highlightbackground=NAVY)
            return
        widget.configure(highlightbackground=BRUSH_EDITOR if n % 2 == 0 else NAVY)
        self.after(140, self._flash, widget, n - 1)

    def _card(self, title, body, legend=False):
        """Step explanation in the explainer bar — auto demo only, not while jumping."""
        if not self.demo_on or self._ff():
            return
        self._explain(title, body, legend=legend)
        self.update_idletasks()

    def _demo_script(self):
        """One visitor round; each yield is a pause in ms, or ("section", i) where
        slide i of DEMO_SECTIONS begins (a jump replays up to that marker)."""
        yield ("section", 0)
        yield from self._demo_click(self.buttons["reset"])
        self.reset()
        self._card("PIXEL BLOCK CHAIN · live demo",
                   "A fragile watermark that shows exactly where an image was tampered "
                   "with. Watch one round, then click anywhere to try it yourself.")
        yield self._bar_read_ms()
        yield ("section", 1)
        self._card("STEP 1 · PROTECT",
                   "We write chains of hash-linked blocks into the least-significant bit "
                   "of every pixel, one independent chain per tile. Invisible: ~51 dB PSNR.")
        # protect while this card is read: one countdown covers the encoding and the rest
        read_ms, t_card = max(self._bar_read_ms(), 10000), time.perf_counter()   # ≥ encode time
        if not self._ff():
            self.pie_span = True
            self._pie(read_ms)
        yield 2500
        yield from self._demo_click(self.buttons["protect"])
        self.protect()
        while self.busy:                            # tile sweep animates meanwhile
            yield 40
        res = self.last_result
        yield max(0, int(read_ms - (time.perf_counter() - t_card) * 1000))   # rest of the card
        yield ("section", 2)
        tw, th = res.width // res.cols, res.height // res.rows
        self._card("ONE CHAIN PER TILE",
                   f"The image is cut into {res.cols}×{res.rows} = {res.cols * res.rows} tiles, "
                   "each hiding its own chain. Labels: tile size → blocks in its chain (e.g. "
                   f"{tw}×{th} px → {res.tile_results[0][0].block_count}). Tampering with one "
                   "tile can never affect its neighbours — guaranteed by design (see Lemma 1 "
                   "in our paper).")
        hold = self._bar_read_ms()
        self._show_grid(res, hold=hold)             # grid stays up while this card is read
        yield hold
        yield ("section", 3)
        self._card("WHAT IT TAKES TO RUN",
                   "Measured tonight on this laptop, one CPU core each, against TrustMark "
                   "(Adobe's neural watermark): PBC embeds about 17× and checks about 3.5× "
                   "faster, with no model, a tenth of the memory and no PyTorch, and it "
                   "shows which tiles changed.")
        if not self._ff():
            self._run_compare()                     # measured bars race in the right pane
        yield max(self._bar_read_ms(), 9000)
        yield ("section", 4)
        self._pane_text(self.right_lbl, "Verification results\nwill appear here")   # drop slide 4's chart
        self._card("STEP 2 · EDIT",
                   "Three edits: a raw scribble, an edit from a PBC-aware editor that records "
                   "itself in the Edit Ledger, and an exact copy of one tile pasted onto "
                   "another.", legend=True)
        # edit while this card is read: one countdown covers the edits and the rest
        read_ms, t_card = max(self._bar_read_ms(), 16000), time.perf_counter()   # ≥ edit time
        if not self._ff():
            self.pie_span = True
            self._pie(read_ms)
        yield 800
        yield from self._demo_click(self.buttons["edit"])
        self.edit()
        yield 1000
        lbl = self.left_lbl
        tw, th = lbl.image.width(), lbl.image.height()
        # magenta raw scribble on the left half, cyan PBC-aware stroke on the right
        # half: kept apart so they never share a tile
        strokes = []                                # preview-space boxes the copy attack avoids
        for compliant, x_lo, x_hi in ((False, 40, tw / 2 - 170), (True, tw / 2 + 20, tw - 200)):
            if compliant:                           # switch pens with a visible click
                yield from self._demo_click(self.pen_btn)
                self._set_pen(True)
                yield 600
            x0, y0 = random.uniform(x_lo, x_hi), random.uniform(50, th - 50)
            length, amp = random.uniform(90, 150), random.uniform(10, 30)
            cycles = random.uniform(1, 2.5)
            strokes.append((x0 - BRUSH_PX, y0 - amp - BRUSH_PX,
                            x0 + length + BRUSH_PX, y0 + amp + BRUSH_PX))
            ox = (lbl.winfo_width() - lbl.image.width()) / 2
            oy = (lbl.winfo_height() - lbl.image.height()) / 2
            rx, ry = lbl.winfo_rootx() + ox, lbl.winfo_rooty() + oy
            yield from self._demo_point(rx + x0, ry + y0)
            t_stroke, done = time.perf_counter(), 0     # the cursor holds the pen, ~1 s per stroke
            while done < 31:
                target = 31 if self._ff() else min(31, 1 + int((time.perf_counter() - t_stroke) * 30))
                while done < target:                    # catch up points if a tick ran late
                    px = x0 + length * done / 30
                    py = y0 + amp * math.sin(2 * math.pi * cycles * done / 30)
                    self._draw(types.SimpleNamespace(x=ox + px, y=oy + py), start=done == 0)
                    done += 1
                self._cursor_to(rx + px, ry + py, pressed=True)
                yield 30
            self._cursor_to(rx + px, ry + py)
            yield 300
        yield from self._demo_copy_tile(strokes)    # third edit: the copy-paste attack
        yield max(0, int(read_ms - (time.perf_counter() - t_card) * 1000))   # rest of the card
        yield ("section", 5)
        self._card("STEP 3 · VERIFY",
                   "The verifier re-reads every chain from the saved file. Each edit leaves "
                   "its own colour; every tile nobody touched stays GREEN.", legend=True)
        # verify while this card is read: the sweep, then the result on the right
        read_ms, t_card = max(self._bar_read_ms(), 10000), time.perf_counter()   # ≥ verify time
        if not self._ff():
            self.pie_span = True
            self._pie(read_ms)
        yield 2500
        yield from self._demo_click(self.buttons["verify"])
        self.do_verify()
        while self.busy:
            yield 40
        yield max(0, int(read_ms - (time.perf_counter() - t_card) * 1000))   # rest of the card
        yield ("section", 6)
        if self.ledger_pick:                        # re-show it: a jump skipped the first showing
            self._show_ledger(*self.ledger_pick)
        self.status.configure(text="Auto demo — click anywhere to pause and try it yourself!")
        yield self._bar_read_ms()                   # verdict + the logged edit's ledger
        yield ("section", 7)
        self._card("EDIT LEDGER",
                   "Each tile's chain records who changed it and how: an operation code "
                   "plus an originator ID, the first 32 bits of SHA-256 of a self-asserted "
                   "name. Binding it to a certificate / Ed25519 signature is future work.")
        yield self._bar_read_ms()
        _, red, green = self._ledger_candidates()
        res = self.verify_result
        copied = [res.tile_results[self.copy_target[1]][self.copy_target[0]]] if self.copy_target else []
        for section, tiles in ((8, green[:1]), (9, red[:1]), (10, copied)):  # logged edit: after VERIFY
            for t in tiles:
                yield ("section", section)
                yield from self._demo_click(point=self._tile_point(t.tx, t.ty))
                self._show_ledger(t.tx, t.ty)
                yield self._bar_read_ms()
        yield 1500


if __name__ == "__main__":
    app = Demo()
    if "--pace" in sys.argv[:-1]:
        app.pace = float(sys.argv[sys.argv.index("--pace") + 1])
    if "--demo" in sys.argv:
        app.after(1500, app.start_demo)
    app.mainloop()
