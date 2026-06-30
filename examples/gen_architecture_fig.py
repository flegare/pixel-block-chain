#!/usr/bin/env python3
"""
PBC Architecture Figure Generator

Produces the system pipeline diagram for Section 3 of the paper.
Saves: output/architecture_overview.png  and  output/architecture_overview.pdf

Usage:
    python examples/gen_architecture_fig.py

MIT License - Copyright (c) 2026 François Légaré
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except ImportError:
        print("matplotlib not available — cannot generate architecture figure")
        return 1

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    # -----------------------------------------------------------------------
    # Colour palette
    # -----------------------------------------------------------------------
    C_BOX   = '#d6eaf8'    # light blue boxes
    C_EDGE  = '#2e86c1'    # box edges
    C_ARROW = '#1a5276'    # arrows
    C_TEXT  = '#1a1a2e'    # labels

    BOX_W, BOX_H = 2.0, 0.85
    ROW_TOP = 4.0    # y-centre of encoding row
    ROW_BOT = 1.8    # y-centre of decoding row

    def box(ax, cx, cy, label, sublabel=""):
        rect = FancyBboxPatch(
            (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.05",
            facecolor=C_BOX, edgecolor=C_EDGE, linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, cy + (0.12 if sublabel else 0), label,
                ha='center', va='center', fontsize=9, fontweight='bold',
                color=C_TEXT, zorder=4)
        if sublabel:
            ax.text(cx, cy - 0.22, sublabel,
                    ha='center', va='center', fontsize=7.5, color='#555',
                    zorder=4)

    def arrow(ax, x0, y0, x1, y1, label=""):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=C_ARROW,
                                   lw=1.5, connectionstyle='arc3,rad=0'))
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my + 0.18, label, ha='center', va='bottom',
                    fontsize=7, color=C_ARROW)

    # -----------------------------------------------------------------------
    # ENCODING row (top)
    # -----------------------------------------------------------------------
    enc_boxes = [
        (1.2,  ROW_TOP, "Input\nImage",      ""),
        (3.4,  ROW_TOP, "Grid\nPartition",   f"cols×rows\n(target ~128 px)"),
        (5.6,  ROW_TOP, "Per-Tile\nGenesis", "SHA-256(oid‖tx‖ty‖W‖H‖t)"),
        (7.8,  ROW_TOP, "Block Chain\nBuild", "CRC + chain hash\nper tile"),
        (10.0, ROW_TOP, "LSB\nEmbedding",    "2 bits/channel\nk=2"),
    ]
    for cx, cy, lbl, sub in enc_boxes:
        box(ax, cx, cy, lbl, sub)

    # arrows between encoding boxes
    for i in range(len(enc_boxes) - 1):
        x0 = enc_boxes[i][0]   + BOX_W / 2
        x1 = enc_boxes[i+1][0] - BOX_W / 2
        y  = ROW_TOP
        arrow(ax, x0, y, x1, y)

    # -----------------------------------------------------------------------
    # DECODING row (bottom)
    # -----------------------------------------------------------------------
    dec_boxes = [
        (10.0, ROW_BOT, "Encoded\nImage",    ""),
        (7.8,  ROW_BOT, "Block\nExtraction", "LSB read\nper tile"),
        (5.6,  ROW_BOT, "Chain\nValidation", "CRC + hash\ncheck"),
        (3.4,  ROW_BOT, "Tile\nAggregation", "GREEN/YELLOW\nRED/ABSENT"),
        (1.2,  ROW_BOT, "Tile\nMap",         "Integrity grid\noutput"),
    ]
    for cx, cy, lbl, sub in dec_boxes:
        box(ax, cx, cy, lbl, sub)

    # arrows between decoding boxes (right-to-left)
    for i in range(len(dec_boxes) - 1):
        x0 = dec_boxes[i][0]   - BOX_W / 2
        x1 = dec_boxes[i+1][0] + BOX_W / 2
        y  = ROW_BOT
        arrow(ax, x0, y, x1, y)

    # vertical connector (encoded image top → encoded image bottom)
    arrow(ax, 10.0, ROW_TOP - BOX_H / 2, 10.0, ROW_BOT + BOX_H / 2,
          label="Encoded\npixels")

    # -----------------------------------------------------------------------
    # Row labels
    # -----------------------------------------------------------------------
    ax.text(0.2, ROW_TOP, "ENCODE", va='center', ha='left', fontsize=10,
            fontweight='bold', color='#145a32',
            rotation=90)
    ax.text(0.2, ROW_BOT, "VERIFY", va='center', ha='left', fontsize=10,
            fontweight='bold', color='#7b241c',
            rotation=90)

    # -----------------------------------------------------------------------
    # Title
    # -----------------------------------------------------------------------
    ax.set_title("Pixel Block Chain (PBC) System Architecture",
                 fontsize=13, fontweight='bold', pad=10, color=C_TEXT)

    fig.tight_layout(pad=0.5)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'architecture')
    os.makedirs(output_dir, exist_ok=True)

    png_path = os.path.join(output_dir, 'architecture_overview.png')
    pdf_path = os.path.join(output_dir, 'architecture_overview.pdf')

    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print()
    print("Copy architecture_overview.pdf to doc/paper/figures/ for LaTeX.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
