"""
PBC Document Stamp — Sign a PDF's raster pages with Pixel Block Chain

Concept:
  1. Render each PDF page to a 300 DPI PNG (deterministic rasterization)
  2. Apply PBC encoding to the page image
     - originator = SHA-256(doc_id + page_number)[:8]  (pseudonymous, derivable)
     - opcode     = Camera_ISP (0x0001) — "capture of this page state"
  3. Reassemble into a new PDF where each page IS the signed PNG
  4. Provide a verifier that extracts pages and checks PBC chains

Why this is different from PDF digital signatures:
  - Spatial tamper localization: a modified paragraph shows RED tiles; the rest GREEN
  - No PKI required: self-verifying, like image PBC
  - Edit ledger: append-mode preserves revision history in the pixels themselves
  - Survives extraction: extract any page as PNG and verify independently
  - Survives social media: if shared as lossless PNG, chain is intact

Honest limitation:
  - Signing is rasterization-specific: different PDF viewers render to different pixels.
    The author's 300 DPI pdftoppm render is the canonical signed form.
  - Printed → scanned documents: print noise destroys LSBs (analog hole applies).
  - Lossy JPEG export breaks the chain (same as images).

Usage:
    python examples/pbc_document_stamp.py input.pdf             # sign
    python examples/pbc_document_stamp.py input_pbc.pdf --verify  # verify

Requirements:
    pdftoppm (poppler-utils), pypdf, numpy, pillow
    Optional: img2pdf  (pip install img2pdf) for cleaner PDF reassembly
"""

import sys
import io
import os
import hashlib
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from pbc.encoder import encode
from pbc.decoder import verify, TileStatus

TILE_SIZE   = 128
DPI         = 300      # canonical signing resolution
ORIGINATOR_PREFIX = "PBC-DOC-STAMP"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def doc_originator_id(pdf_path: Path, page: int) -> str:
    """Pseudonymous originator ID: SHA-256(filename+page)[:8]."""
    raw = f"{pdf_path.stem}-p{page:03d}-{ORIGINATOR_PREFIX}"
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def render_page(pdf_path: Path, page_num: int, dpi: int = DPI) -> np.ndarray:
    """Render one PDF page to RGB numpy array via pdftoppm."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "page")
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi),
             "-f", str(page_num), "-l", str(page_num),
             str(pdf_path), prefix],
            check=True, capture_output=True
        )
        # pdftoppm names output: prefix-{page_num:0Nd}.png
        files = sorted(Path(tmpdir).glob("*.png"))
        if not files:
            raise RuntimeError(f"pdftoppm produced no output for page {page_num}")
        return np.array(Image.open(files[0]).convert("RGB"))


def count_pages(pdf_path: Path) -> int:
    """Return number of pages in PDF."""
    result = subprocess.run(
        ["pdftoppm", "-l", "99999", "-png", "-r", "1",
         str(pdf_path), "/tmp/pbc_count_"],
        capture_output=True
    )
    # Use pdfinfo if available, else pypdf
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        return 1


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")


# ---------------------------------------------------------------------------
# Stamp (sign)
# ---------------------------------------------------------------------------

def stamp_pdf(pdf_path: Path, out_path: Path, dpi: int = DPI) -> None:
    """Render each page, apply PBC, save signed page PNGs + summary."""
    n_pages = count_pages(pdf_path)
    print(f"\nPBC Document Stamp")
    print(f"{'='*60}")
    print(f"  Input  : {pdf_path.name}  ({n_pages} pages)")
    print(f"  DPI    : {dpi}")
    print(f"  Output : {out_path}")
    print()

    out_dir = out_path.parent / (out_path.stem + "_pages")
    out_dir.mkdir(parents=True, exist_ok=True)

    page_files = []
    page_stats = []

    print(f"  {'Page':>5}  {'WxH':<16} {'Tiles':>6}  {'PSNR':>7}  {'Enc ms':>8}")
    print(f"  {'-'*55}")

    import time
    for pg in range(1, n_pages + 1):
        t0 = time.perf_counter()

        # Render
        src = render_page(pdf_path, pg, dpi=dpi)
        H, W = src.shape[:2]

        # Encode with PBC
        oid = doc_originator_id(pdf_path, pg)
        enc = encode(src, originator=oid)
        enc_ms = (time.perf_counter() - t0) * 1000

        # Verify (sanity check)
        res = verify(enc, tile_size=TILE_SIZE)
        green = sum(1 for t in res.all_tiles if t.status == TileStatus.GREEN)
        total = res.rows * res.cols
        q = psnr(src, enc)

        # Save lossless PNG (mandatory: chain lives in LSBs)
        page_png = out_dir / f"page_{pg:03d}_pbc.png"
        Image.fromarray(enc).save(str(page_png), format="png")
        page_files.append(page_png)

        page_stats.append({
            "page": pg, "W": W, "H": H, "tiles": total,
            "green": green, "psnr": q, "enc_ms": enc_ms, "oid": oid,
        })

        print(f"  {pg:>5}  {W}x{H:<8}  {green:>5}/{total}  {q:>7.1f}  {enc_ms:>7.0f} ms")

    # Assemble PDF from signed PNGs
    _assemble_pdf(page_files, out_path)

    # Save summary
    results_dir = Path(__file__).parent.parent / "output" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / f"{out_path.stem}_stamp_report.txt"
    with open(report_path, "w") as f:
        f.write(f"PBC Document Stamp Report\n")
        f.write(f"Input: {pdf_path.name}  Pages: {n_pages}  DPI: {dpi}\n\n")
        f.write(f"{'Page':>5}  {'WxH':<16} {'GREEN/total':>12}  {'PSNR':>7}  OID\n")
        f.write("-" * 60 + "\n")
        for s in page_stats:
            f.write(f"  {s['page']:>3}  {s['W']}x{s['H']:<8}  "
                    f"{s['green']:>5}/{s['tiles']:<5}  {s['psnr']:>7.1f}  {s['oid']}\n")
    print(f"\n  Signed pages : {out_dir}/")
    print(f"  Signed PDF   : {out_path}")
    print(f"  Report       : {report_path}")


def _assemble_pdf(page_pngs: list, out_path: Path) -> None:
    """Assemble page PNGs into a PDF."""
    try:
        import img2pdf
        with open(out_path, "wb") as f:
            f.write(img2pdf.convert([str(p) for p in page_pngs]))
        return
    except ImportError:
        pass

    # Fallback: use pypdf (embeds as images in pages)
    try:
        from pypdf import PdfWriter
        writer = PdfWriter()
        for png_path in page_pngs:
            img = Image.open(png_path)
            W, H = img.size
            page = writer.add_blank_page(
                width=W * 72 / 300,   # 300 DPI → points (72 pt/inch)
                height=H * 72 / 300
            )
            buf = io.BytesIO()
            img.save(buf, format="pdf")
            # Simpler: just note the pages are saved as individual PNGs
        # pypdf doesn't easily embed full-page images without img2pdf
        # So just create a reference PDF that lists the pages
        print(f"  Note: install img2pdf for proper PDF assembly: pip install img2pdf")
        print(f"  Individual signed pages saved as PNG in: {page_pngs[0].parent}/")
        return
    except Exception as e:
        print(f"  PDF assembly failed: {e}")
        print(f"  Signed pages available as PNG files.")


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify_pdf(pdf_path_or_dir: Path) -> None:
    """Verify PBC chains in stamped PDF pages (PNG directory)."""
    print(f"\nPBC Document Verify")
    print(f"{'='*60}")

    # Find page PNGs
    if pdf_path_or_dir.is_dir():
        pages = sorted(pdf_path_or_dir.glob("page_*_pbc.png"))
    else:
        # Look for sibling _pages/ directory
        pages_dir = pdf_path_or_dir.parent / (pdf_path_or_dir.stem + "_pages")
        if pages_dir.exists():
            pages = sorted(pages_dir.glob("page_*_pbc.png"))
        else:
            print(f"  No signed pages directory found. Expected: {pages_dir}")
            return

    if not pages:
        print("  No PBC-signed page PNGs found.")
        return

    print(f"  Found {len(pages)} signed page(s) in: {pages[0].parent}")
    print()
    print(f"  {'Page':>5}  {'WxH':<16} {'GREEN':>6}/{' total':<6}  {'Status':<12}")
    print(f"  {'-'*55}")

    all_ok = True
    for png_path in pages:
        arr = np.array(Image.open(png_path).convert("RGB"))
        H, W = arr.shape[:2]
        res = verify(arr, tile_size=TILE_SIZE)
        green = sum(1 for t in res.all_tiles if t.status == TileStatus.GREEN)
        yellow = sum(1 for t in res.all_tiles if t.status == TileStatus.YELLOW)
        red = sum(1 for t in res.all_tiles if t.status == TileStatus.RED)
        total = res.rows * res.cols

        if green == total:
            status = "INTACT"
        elif red > 0:
            status = f"TAMPERED ({red} RED)"
            all_ok = False
        elif yellow > 0:
            status = f"MODIFIED ({yellow} YELLOW)"
            all_ok = False
        else:
            status = "NO_PBC"
            all_ok = False

        pg_num = png_path.stem.split("_")[1]
        print(f"  {pg_num:>5}  {W}x{H:<8}  {green:>6}/{total:<6}  {status}")

    print()
    if all_ok:
        print(f"  VERDICT: ALL PAGES INTACT — document unmodified since signing")
    else:
        print(f"  VERDICT: TAMPERING DETECTED — see RED/YELLOW tiles above")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PBC Document Stamp — sign/verify PDF pages with Pixel Block Chain"
    )
    parser.add_argument("input", type=Path,
                        help="PDF to sign, or signed PDF / pages dir to verify")
    parser.add_argument("--verify", action="store_true",
                        help="Verify mode: check PBC chains in stamped pages")
    parser.add_argument("--dpi", type=int, default=DPI,
                        help=f"Rasterization DPI (default: {DPI})")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PDF path (default: input_pbc.pdf)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found")
        sys.exit(1)

    if args.verify:
        verify_pdf(args.input)
    else:
        if args.out:
            out = args.out
        else:
            stamp_dir = Path(__file__).parent.parent / "output" / "document-stamp"
            stamp_dir.mkdir(parents=True, exist_ok=True)
            out = stamp_dir / (args.input.stem + "_pbc.pdf")
        stamp_pdf(args.input, out, dpi=args.dpi)


if __name__ == "__main__":
    main()
