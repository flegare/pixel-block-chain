#!/usr/bin/env python3
"""
pbc-cli: Command-line tool for Pixel Block Chain encoding and verification.

Usage:
    python -m pbc encode input.png -o encoded.png --originator "MyCamera"
    python -m pbc verify encoded.png -o report.png
    python -m pbc info encoded.png

MIT License - Copyright (c) 2026 François Légaré
"""

import argparse
import sys
import time
import numpy as np
from PIL import Image


def cmd_encode(args):
    """Encode PBC blocks into an image."""
    from .encoder import encode
    from . import OpCode

    img = np.array(Image.open(args.input).convert('RGB'))
    H, W = img.shape[:2]
    print(f"Encoding PBC into {args.input} ({W}x{H})...")

    from . import compute_grid, DEFAULT_TILE_SIZE
    tile_size = args.tile_size
    cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)
    print(f"  Grid: {cols}x{rows} tiles (~{tile_w}x{tile_h} px each)")

    opcode = getattr(OpCode, args.opcode.upper(), OpCode.CAMERA_ISP)

    start   = time.time()
    encoded = encode(img, originator=args.originator,
                     opcode=opcode, tile_size=tile_size)
    elapsed = time.time() - start

    mse  = np.mean((img.astype(float) - encoded.astype(float)) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')

    output = args.output or args.input.rsplit('.', 1)[0] + '_pbc.png'
    Image.fromarray(encoded).save(output)

    from . import PIXELS_PER_BLOCK
    total_pixels = H * W
    num_blocks   = total_pixels // PIXELS_PER_BLOCK

    print(f"Done in {elapsed:.2f}s")
    print(f"  Total blocks embedded: {num_blocks:,}")
    print(f"  PSNR: {psnr:.1f} dB")
    print(f"  Output: {output}")


def cmd_verify(args):
    """Verify PBC integrity of an image."""
    from .decoder import verify
    from .visualizer import (generate_report_image, generate_overlay,
                              generate_heatmap, render_tile_map)

    img = np.array(Image.open(args.input).convert('RGB'))
    H, W = img.shape[:2]
    print(f"Verifying PBC in {args.input} ({W}x{H})...")

    start  = time.time()
    result = verify(img, strict=args.strict, tile_size=args.tile_size)
    elapsed = time.time() - start

    print(f"Done in {elapsed:.2f}s")
    print()
    print(result.summary())
    print()

    # Tile map summary
    print("Tile integrity map:")
    status_char = {0: 'G', 1: 'Y', 2: 'R', 3: '.'}
    for row in result.tile_results:
        print("  " + " ".join(status_char[t.status] for t in row))
    print("  (G=Intact  Y=Modified  R=Tampered  .=No PBC)")
    print()

    if args.output:
        if args.mode == 'report':
            report = generate_report_image(img, result)
        elif args.mode == 'overlay':
            report = generate_overlay(img, result)
        elif args.mode == 'heatmap':
            report = generate_heatmap(result)
        elif args.mode == 'tilemap':
            import numpy as np
            report = Image.fromarray(render_tile_map(result, cell_size=40))
        else:
            report = generate_report_image(img, result)

        if not isinstance(report, Image.Image):
            report = Image.fromarray(report)
        report.save(args.output)
        print(f"Visualization saved to: {args.output}")

    if args.json:
        import json
        all_tiles = result.all_tiles
        data = {
            'width':  result.width,
            'height': result.height,
            'cols':   result.cols,
            'rows':   result.rows,
            'total_blocks':    result.total_blocks,
            'integrity_score': result.integrity_score,
            'green':  result.green_count,
            'yellow': result.yellow_count,
            'red':    result.red_count,
            'absent': result.absent_count,
            'tiles': [
                {'tx': t.tx, 'ty': t.ty,
                 'status': t.status.name,
                 'block_count': t.block_count}
                for t in all_tiles
            ],
        }
        with open(args.json, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"JSON report saved to: {args.json}")


def cmd_info(args):
    """Quick PBC presence check with grid summary."""
    from .decoder import verify
    from . import compute_grid, DEFAULT_TILE_SIZE

    tile_size = getattr(args, 'tile_size', DEFAULT_TILE_SIZE)
    img = np.array(Image.open(args.input).convert('RGB'))
    H, W = img.shape[:2]

    cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)
    result = verify(img, tile_size=tile_size)

    total = len(result.all_tiles)
    print(f"Image: {W}x{H}")
    print(f"Grid:  {cols}x{rows} tiles (~{tile_w}x{tile_h} px, {total} total)")
    print()

    if result.green_count > 0:
        print(f"PBC DETECTED — {result.integrity_score:.1f}% of tiles intact")
        green_tiles = [t for t in result.all_tiles if t.originator_id]
        if green_tiles:
            oid = green_tiles[0].originator_id
            print(f"  Originator: 0x{oid:08X}")
            from . import OpCode
            first_green_block = next(
                (b for t in result.all_tiles
                 for b in t.blocks if b.status.value == 0), None)
            if first_green_block:
                try:
                    print(f"  Primary Op: {OpCode(first_green_block.opcode).name}")
                except ValueError:
                    print(f"  Primary Op: 0x{first_green_block.opcode:04X}")
    else:
        print("NO PBC DATA DETECTED")


def main():
    from . import DEFAULT_TILE_SIZE

    parser = argparse.ArgumentParser(
        prog='pbc',
        description='Pixel Block Chain — Image Provenance & Integrity Tool'
    )
    sub = parser.add_subparsers(dest='command', help='Command')

    # Encode
    enc = sub.add_parser('encode', help='Encode PBC into an image')
    enc.add_argument('input', help='Input image file')
    enc.add_argument('-o', '--output', help='Output file (default: input_pbc.png)')
    enc.add_argument('--originator', default='pbc-reference-encoder',
                     help='Originator identity string')
    enc.add_argument('--opcode', default='CAMERA_ISP',
                     help='Operation code (e.g., CAMERA_ISP, EDIT_COLOR)')
    enc.add_argument('--tile-size', type=int, default=DEFAULT_TILE_SIZE,
                     dest='tile_size',
                     help=f'Target tile size in pixels (default {DEFAULT_TILE_SIZE})')

    # Verify
    ver = sub.add_parser('verify', help='Verify PBC integrity')
    ver.add_argument('input', help='Input image file')
    ver.add_argument('-o', '--output', help='Output visualization file')
    ver.add_argument('--mode',
                     choices=['report', 'overlay', 'heatmap', 'tilemap'],
                     default='report', help='Visualization mode')
    ver.add_argument('--strict', action='store_true',
                     help='Use strict sync matching (no Hamming tolerance)')
    ver.add_argument('--json', help='Save JSON report to file')
    ver.add_argument('--tile-size', type=int, default=DEFAULT_TILE_SIZE,
                     dest='tile_size',
                     help=f'Target tile size used at encoding (default {DEFAULT_TILE_SIZE})')

    # Info
    inf = sub.add_parser('info', help='Quick PBC presence check')
    inf.add_argument('input', help='Input image file')
    inf.add_argument('--tile-size', type=int, default=DEFAULT_TILE_SIZE,
                     dest='tile_size',
                     help=f'Target tile size (default {DEFAULT_TILE_SIZE})')

    args = parser.parse_args()

    if args.command == 'encode':
        cmd_encode(args)
    elif args.command == 'verify':
        cmd_verify(args)
    elif args.command == 'info':
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
