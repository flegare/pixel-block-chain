"""
Pixel Block Chain (PBC) - Integrity Visualizer

Generates tile-level integrity maps and composite report images.

MIT License - Copyright (c) 2026 François Légaré
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional

from .decoder import (GridResult, TileResult, TileStatus,
                      BlockResult, BlockStatus)

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

# Tile-status colours used in the tile map
TILE_COLORS = {
    TileStatus.GREEN:  (46,  204, 113),   # #2ecc71
    TileStatus.YELLOW: (243, 156,  18),   # #f39c12
    TileStatus.RED:    (231,  76,  60),   # #e74c3c
    TileStatus.ABSENT: (149, 165, 166),   # #95a5a6
}

# Semi-transparent overlay colours (R, G, B, A)
TILE_COLORS_RGBA = {
    TileStatus.GREEN:  (46,  204, 113, 100),
    TileStatus.YELLOW: (243, 156,  18,  130),
    TileStatus.RED:    (231,  76,  60,  160),
    TileStatus.ABSENT: (149, 165, 166, 100),
}

TILE_LABELS = {
    TileStatus.GREEN:  "INTACT",
    TileStatus.YELLOW: "MODIFIED",
    TileStatus.RED:    "TAMPERED",
    TileStatus.ABSENT: "NO PBC",
}

# Keep backward-compat aliases using BlockStatus so existing callers compile
STATUS_COLORS_SOLID = {
    BlockStatus.GREEN:  (46,  204, 113),
    BlockStatus.YELLOW: (243, 156,  18),
    BlockStatus.RED:    (231,  76,  60),
    BlockStatus.ABSENT: (149, 165, 166),
}

STATUS_LABELS = {
    BlockStatus.GREEN:  "GREEN – Intact",
    BlockStatus.YELLOW: "YELLOW – Re-encoded",
    BlockStatus.RED:    "RED – Tampered",
    BlockStatus.ABSENT: "ABSENT – No PBC",
}


# =============================================================================
# Tile Map (primary output)
# =============================================================================

def render_tile_map(grid_result: GridResult,
                    cell_size: int = 40,
                    show_labels: bool = True) -> np.ndarray:
    """
    Render a compact tile integrity map.

    Each tile is represented as one coloured cell:
      GREEN  (#2ecc71) — intact
      YELLOW (#f39c12) — re-encoded (PBC-aware edit)
      RED    (#e74c3c) — tampered (raw pixel modification)
      ABSENT (#95a5a6) — no PBC data

    Args:
        grid_result: Output of decoder.verify().
        cell_size:   Pixel size of each tile cell in the map (default 40).
        show_labels: If True, draw (tx,ty) coordinate text in each cell.

    Returns:
        RGB numpy array of shape (rows*cell_size, cols*cell_size, 3).
    """
    rows = grid_result.rows
    cols = grid_result.cols
    h    = rows * cell_size
    w    = cols * cell_size

    img  = np.zeros((h, w, 3), dtype=np.uint8)

    for ty in range(rows):
        for tx in range(cols):
            tile   = grid_result.tile_results[ty][tx]
            color  = TILE_COLORS[tile.status]
            y0, y1 = ty * cell_size, (ty + 1) * cell_size
            x0, x1 = tx * cell_size, (tx + 1) * cell_size
            img[y0:y1, x0:x1] = color
            # 1-pixel black grid lines
            img[y0, x0:x1] = (0, 0, 0)
            img[y0:y1, x0] = (0, 0, 0)

    pil = Image.fromarray(img, 'RGB')

    if show_labels and cell_size >= 20:
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                max(8, cell_size // 5))
        except (OSError, IOError):
            font = ImageFont.load_default()

        for ty in range(rows):
            for tx in range(cols):
                label = f"{tx},{ty}"
                cx    = tx * cell_size + cell_size // 2
                cy    = ty * cell_size + cell_size // 2
                draw.text((cx - cell_size // 4, cy - cell_size // 8),
                          label, fill=(255, 255, 255), font=font)

    return np.array(pil)


# =============================================================================
# Overlay on original image
# =============================================================================

def generate_overlay(image: np.ndarray,
                     result: GridResult,
                     opacity: float = 0.4) -> Image.Image:
    """
    Generate a semi-transparent tile integrity overlay on the original image.

    Each tile region is coloured according to its TileStatus.

    Args:
        image:   Original RGB image (H, W, 3), uint8.
        result:  GridResult from decoder.verify().
        opacity: Overlay opacity (0.0 = invisible, 1.0 = opaque).

    Returns:
        PIL Image with tile integrity overlay.
    """
    H, W = image.shape[:2]
    base    = Image.fromarray(image, 'RGB').convert('RGBA')
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ov_arr  = np.array(overlay)

    cols, rows = result.cols, result.rows
    tile_w = W // cols
    tile_h = H // rows

    for ty in range(rows):
        for tx in range(cols):
            tile  = result.tile_results[ty][tx]
            r, g, b, a = TILE_COLORS_RGBA[tile.status]
            alpha = int(a * opacity / 0.4)

            x0 = tx * tile_w
            x1 = (tx + 1) * tile_w if tx < cols - 1 else W
            y0 = ty * tile_h
            y1 = (ty + 1) * tile_h if ty < rows - 1 else H

            ov_arr[y0:y1, x0:x1] = (r, g, b, alpha)

    overlay   = Image.fromarray(ov_arr, 'RGBA')
    composite = Image.alpha_composite(base, overlay)
    return composite.convert('RGB')


# =============================================================================
# Block-level heatmap (unchanged, kept for backward compat / demo)
# =============================================================================

def generate_heatmap(result: GridResult, scale: int = 1) -> Image.Image:
    """
    Generate a standalone tile-colour heatmap at image resolution.

    Each pixel is coloured with the status of the tile it belongs to.
    """
    H, W = result.height, result.width
    heatmap = np.zeros((H, W, 3), dtype=np.uint8)

    cols, rows = result.cols, result.rows
    tile_w = W // cols
    tile_h = H // rows

    for ty in range(rows):
        for tx in range(cols):
            tile  = result.tile_results[ty][tx]
            color = TILE_COLORS[tile.status]
            x0 = tx * tile_w
            x1 = (tx + 1) * tile_w if tx < cols - 1 else W
            y0 = ty * tile_h
            y1 = (ty + 1) * tile_h if ty < rows - 1 else H
            heatmap[y0:y1, x0:x1] = color

    img = Image.fromarray(heatmap, 'RGB')
    if scale > 1:
        img = img.resize((W * scale, H * scale), Image.NEAREST)
    return img


# =============================================================================
# Block grid (compact view — now shows tiles, not individual blocks)
# =============================================================================

def generate_block_grid(result: GridResult,
                        grid_width: Optional[int] = None) -> Image.Image:
    """
    Generate a compact grid visualization — one cell per tile.
    """
    all_tiles = result.all_tiles
    n = len(all_tiles)
    if n == 0:
        return Image.new('RGB', (100, 100), (128, 128, 128))

    if grid_width is None:
        grid_width = result.cols

    grid_height = result.rows
    cell_size   = max(4, min(24, 800 // max(grid_width, 1)))

    img_w = grid_width  * cell_size
    img_h = grid_height * cell_size
    grid  = np.full((img_h, img_w, 3), 40, dtype=np.uint8)

    for tile in all_tiles:
        col   = tile.tx
        row   = tile.ty
        color = TILE_COLORS[tile.status]
        y0 = row * cell_size + 1
        y1 = (row + 1) * cell_size - 1
        x0 = col * cell_size + 1
        x1 = (col + 1) * cell_size - 1
        grid[y0:y1, x0:x1] = color

    return Image.fromarray(grid, 'RGB')


# =============================================================================
# Full Report Image
# =============================================================================

def generate_report_image(image: np.ndarray,
                          result: GridResult) -> Image.Image:
    """
    Generate a comprehensive report image: overlay + tile map + stats panel.

    Args:
        image:  Original RGB image (H, W, 3), uint8.
        result: GridResult from decoder.verify().

    Returns:
        PIL Image with full visual report.
    """
    H, W = image.shape[:2]

    overlay  = generate_overlay(image, result, opacity=0.45)
    tile_map = Image.fromarray(render_tile_map(result, cell_size=40))

    # Canvas: overlay left | stats right
    panel_w  = max(320, W // 3)
    canvas_w = W + panel_w
    canvas_h = max(H + tile_map.height + 10, 500)
    canvas   = Image.new('RGB', (canvas_w, canvas_h), (30, 30, 35))

    # Paste overlay (top-left)
    canvas.paste(overlay.convert('RGB'), (0, 0))

    # Paste tile map below overlay
    tm_y = H + 5
    if tm_y + tile_map.height <= canvas_h:
        canvas.paste(tile_map, (0, tm_y))

    # Stats panel
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_body  = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_body  = font_title
        font_small = font_title

    x0 = W + 20
    y  = 20

    draw.text((x0, y), "PBC Integrity Report",
              fill=(255, 255, 255), font=font_title)
    y += 35

    draw.text((x0, y), f"Image: {W}\u00d7{H} ({W*H:,} px)",
              fill=(180, 180, 180), font=font_body)
    y += 22
    draw.text((x0, y), f"Grid:  {result.cols}\u00d7{result.rows} tiles",
              fill=(180, 180, 180), font=font_body)
    y += 22
    draw.text((x0, y), f"Blocks: {result.total_blocks:,}",
              fill=(180, 180, 180), font=font_body)
    y += 35

    total  = len(result.all_tiles)
    bar_w  = panel_w - 40
    bar_h  = 22
    status_rows = [
        (TileStatus.GREEN,  result.green_count,  "INTACT"),
        (TileStatus.YELLOW, result.yellow_count, "MODIFIED"),
        (TileStatus.RED,    result.red_count,    "TAMPERED"),
        (TileStatus.ABSENT, result.absent_count, "NO PBC"),
    ]

    for status, count, label in status_rows:
        pct   = count / total * 100 if total else 0
        color = TILE_COLORS[status]
        draw.text((x0, y), f"{label}", fill=color, font=font_body)
        y += 20
        draw.rectangle([x0, y, x0 + bar_w, y + bar_h], fill=(60, 60, 65))
        fill_w = int(bar_w * pct / 100)
        if fill_w > 0:
            draw.rectangle([x0, y, x0 + fill_w, y + bar_h], fill=color)
        draw.text((x0 + bar_w + 5, y + 3), f"{pct:.1f}%",
                  fill=(200, 200, 200), font=font_small)
        y += bar_h + 15

    y += 10
    score = result.integrity_score
    score_color = (TILE_COLORS[TileStatus.GREEN]  if score > 95 else
                   TILE_COLORS[TileStatus.YELLOW] if score > 50 else
                   TILE_COLORS[TileStatus.RED])
    draw.text((x0, y), f"Integrity: {score:.1f}%",
              fill=score_color, font=font_title)
    y += 30

    status_text = ("HIGH INTEGRITY" if score > 95 else
                   "PARTIAL INTEGRITY" if score > 50 else
                   "LOW INTEGRITY" if result.green_count > 0 else
                   "NO PBC DATA")
    draw.text((x0, y), status_text, fill=score_color, font=font_body)

    # Originator from first GREEN tile
    y += 40
    green_tiles = [t for t in result.all_tiles
                   if t.status == TileStatus.GREEN and t.originator_id]
    if green_tiles:
        oid = green_tiles[0].originator_id
        draw.text((x0, y), f"Originator: 0x{oid:08X}",
                  fill=(180, 180, 180), font=font_small)

    return canvas
