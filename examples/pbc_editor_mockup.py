"""
Generate a publication-quality mockup of a PBC-aware image editor interface.

The figure shows the kind of UI that edit_ledger_demo.py would drive:
  - Main image area with per-tile integrity overlay
  - Tile-status grid panel
  - Edit Ledger sidebar listing per-tile block history
  - Toolbar strip at top

Output: pbc-documents/paper/figures/fig_editor_mockup.png
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT   = Path(__file__).resolve().parents[1]
LEDGER_DIR = PROJECT / "output" / "edit-ledger"
OVERLAY_IMG = LEDGER_DIR / "ledger_07_overlay_final.png"
TILEMAP_IMG = LEDGER_DIR / "ledger_08_tilemap_final.png"
OUT_DIR   = PROJECT.parent / "pbc-documents" / "paper" / "figures"
OUT_FILE  = OUT_DIR / "fig_editor_mockup.png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
TOOLBAR_BG = "#2d2d2d"
PANEL_BG   = "#1e1e1e"
STATUS_BG  = "#252526"
TEXT_LIGHT = "#cccccc"
TEXT_DIM   = "#888888"
GREEN_TILE = "#00c853"
YELLOW_TILE= "#ffab00"
RED_TILE   = "#d50000"
BLUE_ACCENT= "#007acc"
BORDER_COL = "#3c3c3c"

# ---------------------------------------------------------------------------
# Load source images
# ---------------------------------------------------------------------------
overlay_np = np.array(Image.open(OVERLAY_IMG).convert("RGB"))
tilemap_np = np.array(Image.open(TILEMAP_IMG).convert("RGB"))

H, W = overlay_np.shape[:2]

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 8.5), facecolor=PANEL_BG)

# ── Layout: [toolbar(top)] [image(left) | sidebar(right)] [statusbar(bottom)]
# Use GridSpec: 3 rows × 2 cols, with header/footer rows thin
gs = fig.add_gridspec(3, 2,
                      height_ratios=[0.055, 0.88, 0.065],
                      width_ratios=[0.65, 0.35],
                      left=0.0, right=1.0, top=1.0, bottom=0.0,
                      hspace=0.0, wspace=0.005)

# ── Toolbar (spans full width) ──────────────────────────────────────────────
ax_tool = fig.add_subplot(gs[0, :])
ax_tool.set_facecolor(TOOLBAR_BG)
ax_tool.set_xlim(0, 1)
ax_tool.set_ylim(0, 1)
ax_tool.axis("off")

# Title bar text
ax_tool.text(0.01, 0.55, "PBC Editor  —  ledger_07_overlay_final.png",
             color=TEXT_LIGHT, fontsize=9, va="center", fontfamily="monospace")
ax_tool.text(0.99, 0.55, "PBC v5  |  Ed25519  |  flegare@gmail.com",
             color=TEXT_DIM, fontsize=7.5, va="center", ha="right",
             fontfamily="monospace")

# Fake toolbar buttons
buttons = ["File", "Edit", "View", "PBC", "Verify", "Help"]
for i, label in enumerate(buttons):
    ax_tool.text(0.01 + i * 0.055, 0.1, label,
                 color=TEXT_LIGHT, fontsize=7.5, va="bottom",
                 fontfamily="sans-serif")

# ── Main image area ──────────────────────────────────────────────────────────
ax_img = fig.add_subplot(gs[1, 0])
ax_img.set_facecolor(PANEL_BG)
ax_img.imshow(overlay_np, interpolation="bilinear")
ax_img.set_xticks([])
ax_img.set_yticks([])
for spine in ax_img.spines.values():
    spine.set_edgecolor(BORDER_COL)
    spine.set_linewidth(0.8)

# ── Sidebar: Tile Map + Edit Ledger ─────────────────────────────────────────
ax_side = fig.add_subplot(gs[1, 1])
ax_side.set_facecolor(STATUS_BG)
ax_side.set_xlim(0, 1)
ax_side.set_ylim(0, 1)
ax_side.axis("off")

# Section header: Tile Map
ax_side.text(0.05, 0.975, "Tile Integrity Map", color=TEXT_LIGHT,
             fontsize=8.5, fontweight="bold", va="top")
ax_side.axhline(0.965, color=BORDER_COL, linewidth=0.8)

# Embed tilemap image in the sidebar (top ~38%)
ax_tmap = ax_side.inset_axes([0.03, 0.63, 0.94, 0.32])
ax_tmap.imshow(tilemap_np, interpolation="nearest")
ax_tmap.set_xticks([])
ax_tmap.set_yticks([])
for spine in ax_tmap.spines.values():
    spine.set_edgecolor(BORDER_COL)
    spine.set_linewidth(0.6)

# Section header: Edit Ledger
ax_side.axhline(0.605, color=BORDER_COL, linewidth=0.8)
ax_side.text(0.05, 0.598, "Edit Ledger  (tile [2,1])", color=TEXT_LIGHT,
             fontsize=8.5, fontweight="bold", va="top")
ax_side.axhline(0.588, color=BORDER_COL, linewidth=0.5)

# Simulated ledger entries for tile (2,1)
ledger_entries = [
    ("Blk 0", "Camera",  "Genesis",         "#0288d1"),
    ("Blk 1", "Alice",   "Edit_Color",       GREEN_TILE),
    ("Blk 2", "Alice",   "Edit_Crop_Safe",   GREEN_TILE),
    ("Blk 3", "Bob",     "Edit_Composite",   "#ff9800"),
    ("Blk 4", "Bob",     "Export_Lossless",  GREEN_TILE),
]
y0 = 0.57
row_h = 0.085
for blk, author, op, col in ledger_entries:
    y = y0 - ledger_entries.index((blk, author, op, col)) * row_h
    # Color dot
    ax_side.add_patch(plt.Circle((0.06, y + 0.025), 0.018,
                                  color=col, transform=ax_side.transData))
    ax_side.text(0.11, y + 0.04, blk, color=TEXT_DIM, fontsize=6.8,
                 va="center", fontfamily="monospace")
    ax_side.text(0.28, y + 0.04, author, color=TEXT_LIGHT, fontsize=7.2,
                 va="center", fontweight="bold")
    ax_side.text(0.28, y + 0.005, op, color=TEXT_DIM, fontsize=6.5,
                 va="center", fontfamily="monospace")

# Stats footer in sidebar
ax_side.axhline(0.155, color=BORDER_COL, linewidth=0.8)
stats = [
    ("Total tiles", "40 / 40 GREEN"),
    ("Authors",     "Camera · Alice · Bob"),
    ("Blocks/tile", "up to 5"),
    ("PBC version", "5  |  Ed25519"),
]
y_s = 0.145
for label, val in stats:
    ax_side.text(0.05, y_s, f"{label}:", color=TEXT_DIM, fontsize=6.5, va="top")
    ax_side.text(0.42, y_s, val, color=TEXT_LIGHT, fontsize=6.5, va="top",
                 fontfamily="monospace")
    y_s -= 0.033

# ── Status bar (bottom, spans full width) ───────────────────────────────────
ax_status = fig.add_subplot(gs[2, :])
ax_status.set_facecolor(BLUE_ACCENT)
ax_status.set_xlim(0, 1)
ax_status.set_ylim(0, 1)
ax_status.axis("off")

status_parts = [
    (0.008, "✓ All 40 tiles INTACT"),
    (0.22,  "Originator: flegare@gmail.com (0x8A2B9AF0)"),
    (0.60,  "3 authors · 978 × 678 px · 8×5 grid"),
    (0.88,  "PBC v5 — MIT"),
]
for x, txt in status_parts:
    ax_status.text(x, 0.48, txt, color="white", fontsize=7.2, va="center",
                   fontfamily="monospace")

# ── Finalize ─────────────────────────────────────────────────────────────────
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight",
            facecolor=PANEL_BG, pad_inches=0.01)
plt.close(fig)
print(f"Editor mockup saved: {OUT_FILE}")
