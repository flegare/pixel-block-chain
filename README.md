# Pixel Block Chain (PBC)

**Spatial Tamper Localization and Crop-Resilient Edit Ledger for Images in the Wild**

Reference implementation for the paper by François Légaré, Sion Israel Sion, and
Alain April (École de technologie supérieure), IEEE ICIP 2026.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

PBC embeds tamper-evident provenance directly into image pixel data using LSB steganography — structured as an independent chain of blocks per spatial tile. Unlike metadata-based systems (EXIF, C2PA), PBC provenance lives in the pixel values themselves and **survives metadata stripping** by social media platforms, messaging apps, and screenshot tools.

---

## What PBC Does

| Capability | Status |
|---|---|
| Detects pixel-level tampering | ✅ validated |
| Localizes tampered regions spatially (tile-level) | ✅ validated |
| No cascade: tampering in one tile leaves others intact | ✅ validated |
| Tracks edit types and history per region (Edit Ledger) | ✅ validated |
| Per-region multi-author attribution (no external DB) | ✅ validated |
| Survives metadata stripping | ✅ validated |
| Crop-resilient mode (PBC-Forest): ~43% survival after 60×80% crop | ✅ validated |
| Video: detect frame tamper / reorder / insert / delete | ✅ validated |
| Document stamp: per-page QR binding content + page order | ✅ validated (26 pages, 100% PASS) |
| Visible glyph watermark: human-perceivable provenance cue | ✅ validated (99% accuracy @ PSNR 22 dB) |
| Proves image depicts reality | ❌ (no provenance system can) |
| Defeats the analog hole | ❌ (shared limitation with C2PA) |
| Works after JPEG compression (k=1) | ❌ (JPEG destroys all LSBs; lossless formats required) |
| Proves originator identity without external trust layer | ❌ (see [Trust Model](#trust-model)) |

---

## Installation

```bash
git clone https://github.com/flegare/pixel-block-chain.git
cd pixel-block-chain
pip install numpy Pillow
```

Additional dependencies for specific examples:

```bash
pip install scipy          # benchmark plots
pip install qrcode zxingcpp  # margin QR stamp examples
pip install rawpy          # camera RAW workflow
pip install pyhanko pyhanko-certvalidator  # PDF signing
```

---

## Quick Start

### Encode PBC into an image

```python
import numpy as np
from PIL import Image
from pbc.encoder import encode

img = np.array(Image.open("photo.png").convert("RGB"))
encoded = encode(img, originator="MyCamera-SN12345", opcode=0x0001)
Image.fromarray(encoded).save("photo_pbc.png")
# Save as PNG/TIFF/WebP-lossless — JPEG destroys the chain
```

### Verify integrity

```python
from pbc.decoder import verify
from pbc.visualizer import generate_report_image

result = verify(encoded)
print(result.summary())
report = generate_report_image(encoded, result)
report.save("report.png")
```

### CLI

```bash
python -m pbc encode photo.png -o photo_pbc.png --originator "MyCamera-SN12345"
python -m pbc verify photo_pbc.png -o report.png
python -m pbc info photo_pbc.png
```

### Full demo

```bash
python examples/demo.py
```

---

## Architecture

### Grid of Independent Tile Chains

PBC divides the image into an adaptive grid of tiles (default 128×128 px). Each tile gets its own independent block chain with its own genesis hash. Tampering in one tile **never cascades** to others — each tile is independently verifiable.

```
Image (W × H)
┌────┬────┬────┬────┐
│ T  │ T  │ T  │ T  │   Each tile = independent block chain
├────┼────┼────┼────┤   Tile genesis = SHA-256(oid ‖ tx ‖ ty ‖ ts)[0:48]
│ T  │ 🔴 │ T  │ T  │
├────┼────┼────┼────┤   One tampered tile → only that tile RED
│ T  │ T  │ T  │ T  │   All other tiles remain independently GREEN
└────┴────┴────┴────┘
```

### Block Structure (256 bits = 32 bytes)

```
┌────────┬─────────┬──────────────┬────────┬─────────┬────────┬────────┬───────────┬───────┬────────────┐
│  Sync  │ Version │ Originator   │ OpCode │  Block  │ Tile X │ Tile Y │ Timestamp │ CRC16 │ Chain Hash │
│ 48 bit │  8 bit  │   32 bit     │ 16 bit │  16 bit │  8 bit │  8 bit │  24 bit   │16 bit │  48 bit    │
└────────┴─────────┴──────────────┴────────┴─────────┴────────┴────────┴───────────┴───────┴────────────┘
```

Each pixel carries 3 bits (1 LSB × 3 channels), so each block spans ~86 pixels.
A 128×128 tile holds ~190 blocks. A 12 MP image has ~750 tiles and ~142,500 blocks.

### Chain Hash

```
Block[n].chain_hash = SHA-256(Block[n-1])[0:48 bits]
Block[0].chain_hash = SHA-256(oid ‖ tile_x ‖ tile_y ‖ timestamp)[0:48 bits]
```

Modification of any pixel in block n changes that block's bytes → invalidates block n+1's stored chain hash → detected as YELLOW or RED. The failure is **local to the tile**.

### Verification States

| State | Meaning |
|---|---|
| 🟢 **GREEN / INTACT** | CRC valid, chain hash valid — provenance intact |
| 🟡 **YELLOW / MODIFIED** | CRC valid, chain broken — PBC-aware re-encoding |
| 🔴 **RED / TAMPERED** | CRC invalid — raw pixel modification detected |
| ⬜ **ABSENT / NO_PBC** | No sync frame found — no provenance embedded |

---

## Edit Ledger and Multi-Author Attribution

PBC-aware editors operate in **append mode**: rather than overwriting the existing chain, they append new blocks with their own Originator ID and operation code. The ordered sequence of blocks in each tile constitutes the **Edit Ledger** — a tamper-evident, chronological record of every transformation from RAW capture to present:

```
Tile (2,1) Edit Ledger:
  Block 0: oid=NikonZ9-SN2024  op=Camera_ISP    t=T+0s
  Block 1: oid=Lightroom-Alice  op=Edit_Color    t=T+3600s
  Block 2: oid=Photoshop-Bob   op=Edit_Retouch  t=T+7200s
```

This records spatial attribution without any external database.

### Operation Code Registry

| Code | Name | Description |
|------|------|-------------|
| `0x0000` | `Camera_Raw` | Unprocessed sensor capture |
| `0x0001` | `Camera_ISP` | Camera ISP pipeline output |
| `0x0010` | `Edit_Crop` | Crop by PBC-aware editor |
| `0x0011` | `Edit_Color` | Color/exposure adjustment |
| `0x0012` | `Edit_Resize` | Resample/rescale |
| `0x0020` | `Edit_Retouch` | Healing, clone stamp |
| `0x0030` | `Edit_AI_Enh` | AI super-res, denoise |
| `0x0031` | `Edit_AI_Gen` | AI-generated inpainting |
| `0x0040` | `Export_Compress` | Lossy compression applied |
| `0x0060` | `Batch_Tonal` | N tonal/parametric edits condensed |
| `0x0061` | `Batch_Structural` | N structural edits condensed |
| `0xFFFE` | `Chain_Repair` | Re-encoded for chain repair |

---

## Extensions

### PBC-Forest (Crop Resilience)

Standard grid mode fails at non-aligned crop boundaries. PBC-Forest places every block as its own independent genesis block at a pseudo-random position. Any surviving block independently authenticates origin:

```
Results (leo.jpg, 60%×80% non-aligned crop):
  Grid mode:        0.0% survival
  Single-chain scatter: ≤0.5% survival
  PBC-Forest:      41–45% survival  (theoretical max: 47.9%)
```

```bash
python examples/forest_scatter_test.py
```

### Video PBC

Each frame's tile grid is anchored to the terminal block bytes of the same tile in the previous frame — creating per-tile inter-frame chains. Detects pixel tampering, frame swaps, frame insertion, and frame deletion with spatial precision.

```bash
python examples/video_pbc.py
```

### Margin QR Document Stamp

One 220×220 px QR code per page (bottom-right margin) carries:
- `content_hash`: SHA-256 of the binarized page (light-insensitive)
- `chain_hash`: SHA-256 of the previous page's rendered QR pixel array

Validated on 26 pages: 100% PASS. Detects content modification, page removal, and page reordering.

```bash
python examples/pbc_margin_qr_stamp.py
```

### Visible Glyph Watermark

Genesis-hash-derived 44×44 px glyph overlaid at configurable opacity. Detected via cross-correlation — robust to moderate brightness and contrast changes.

| Opacity α | PSNR | Accuracy |
|---|---|---|
| 0.35 | 24.4 dB | 94.8% |
| **0.45** | **22.1 dB** | **99.0%** |
| 0.65 | 19.0 dB | 99.7% |

```bash
python examples/pbc_glyph_watermark_demo.py
```

---

## Format Robustness

| Format | Type | Chain Survives |
|---|---|---|
| PNG | lossless | ✅ |
| BMP | lossless | ✅ |
| TIFF (LZW / raw) | lossless | ✅ |
| WebP (lossless) | lossless | ✅ (most compact: 0.69× PNG) |
| JPEG (any quality) | lossy | ❌ |
| WebP (lossy) | lossy | ❌ |

JPEG destroys LSB data through YCbCr color-space rounding, regardless of quality setting. **Always save PBC-encoded images in a lossless format.**

---

## Performance

Single-threaded Python on AMD Ryzen consumer hardware:

| Resolution | Tiles | Encode | Verify | PSNR |
|---|---|---|---|---|
| 256² | 4 | 91 ms | 87 ms | 51.2 dB |
| 1024² | 64 | 2.0 s | 1.9 s | 51.2 dB |
| 12 MP (4032×3024) | 768 | 19.3 s | 19.1 s | 51.2 dB |
| 24 MP (6000×4000) | 1457 | 34.2 s | 34.9 s | 51.2 dB |
| 61 MP (9504×6336) | 3700 | 100 s | 99 s | 51.2 dB |

Throughput scales linearly (O(n)) with pixel count at ~0.65 MP/s.
PSNR is constant at 51.2 dB regardless of image size.

```bash
python examples/benchmark.py
python examples/highres_benchmark.py
```

---

## Reviewer Reproducibility

The paper-critical claims can be checked from a fresh clone with only the core
runtime dependencies plus `pytest`:

```bash
git clone https://github.com/flegare/pixel-block-chain.git
cd pixel-block-chain
python -m pip install -e .
python -m pip install pytest
python -m pytest -q
python tools/validate_paper_claims.py
```

The validation runner writes:

- `results/paper_validation.json` — machine-readable metrics and thresholds.
- `results/paper_validation.md` — compact human-readable summary.

Stable paper figures can be regenerated from the public implementation:

```bash
python tools/regenerate_paper_figures.py
```

To write figures into an external paper folder:

```bash
python tools/regenerate_paper_figures.py --output-dir path/to/paper/figures
```

Dataset provenance for the reviewer validation is tracked in
`results/dataset_manifest.json`. The full 99-image paper benchmark uses 94 real
images (60 COCO, 12 Oxford Flowers, 21 Oxford-IIIT Pets, and `leo.jpg`) plus 5
deterministic synthetic probes. The manifest records exact filenames,
dimensions, byte sizes, and SHA-256 hashes for the real-image benchmark set.

The CI workflow in `.github/workflows/validation.yml` runs the unit tests and
paper validation runner on every push and pull request.

---

## Trust Model

PBC's Originator ID is a 32-bit truncated SHA-256 of a self-asserted identity string. This is a **persistent pseudonymous fingerprint**, not a verified identity.

- **Without external trust layer:** PBC answers *"has this image been modified since encoding, and where?"*
- **With Optional Trust Layer (PKI/transparency log):** Originator ID bound to an X.509 certificate answers *"was the encoder a trusted device?"*

PBC is designed to **complement C2PA**, not replace it. C2PA provides origin authentication via PKI; PBC provides spatial integrity that persists where C2PA's file-level metadata cannot (social media re-encoding, metadata stripping).

---

## Paper

> François Légaré, Sion Israel Sion, and Alain April. *Pixel Block Chain: Spatial
> Tamper Localization and Crop-Resilient Edit Ledger for Images in the Wild.*
> IEEE International Conference on Image Processing (ICIP), 2026.

The camera-ready paper (`paper/PBC_ICIP2026_CameraReady.pdf`), its LaTeX source
(`paper/PBC_ICIP2026_CameraReady.tex`), and all figures (`paper/figures/`) are
included. The PDF compiles from the source with a standard LaTeX distribution
(`pdflatex`, IEEEtran class).

---

## Repository Structure

```
pixel-block-chain/
├── pbc/                    Core library
│   ├── encoder.py          LSB block encoding, grid and forest modes
│   ├── decoder.py          Block extraction, chain verification
│   ├── scatter.py          PBC-Forest scatter placement
│   ├── video.py            Video inter-frame chain
│   ├── visualizer.py       Tile integrity map visualization
│   └── cli.py              Command-line interface
├── examples/               Demonstration and experiment scripts
│   ├── demo.py             Core encode/verify/tamper scenarios
│   ├── edit_ledger_demo.py Multi-author Edit Ledger experiment
│   ├── forest_scatter_test.py  Crop survivability (PBC-Forest)
│   ├── video_pbc.py        Video inter-frame chain validation
│   ├── pbc_margin_qr_stamp.py  Document stamp (26-page validation)
│   ├── pbc_glyph_watermark_demo.py  Visible glyph watermark
│   ├── benchmark.py        Performance benchmarks
│   ├── highres_benchmark.py  Camera-grade resolution benchmarks
│   ├── multi_image_eval.py  Generalization across 99 images
│   └── img/leo.jpg         Primary test image
├── paper/
│   ├── PBC_ICIP2026_CameraReady.pdf   Camera-ready paper (ICIP 2026)
│   ├── PBC_ICIP2026_CameraReady.tex   LaTeX source
│   └── figures/            All paper figures
├── results/                Experiment result files (text/CSV)
├── web/
│   ├── pbc_verifier.html   Browser-based verifier UI
│   └── https_server.py     Local HTTPS server for the verifier
├── setup.py
├── LICENSE                 MIT
└── README.md
```

---

## Contributing

Contributions are welcome. Open areas:

- **ECC layer**: Reed-Solomon or BCH across the 256-bit block payload — would enable JPEG robustness
- **WebAssembly build**: Browser-side verification without a server
- **C99 library** (`libpbc-core`): Firmware-deployable implementation for camera ISPs
- **Perceptual content hash**: Replace SHA-256 in margin QR `content_hash` to support camera-captured page verification
- **Audio PBC**: Extend the LSB chain to WAV/FLAC audio segments

Please open an issue before starting significant work to avoid duplication.

Before publishing a fork, release archive, or reviewer snapshot, run:

```bash
python tools/sanitize_for_publication_git.py
```

See `PUBLICATION_CHECKLIST.md` for the public-release checklist.

---

## License

MIT License — Copyright (c) 2026 François Légaré. See [LICENSE](LICENSE).
