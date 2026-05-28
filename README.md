# Pixel Block Chain (PBC)

**Spatial Tamper Localization and Crop-Resilient Provenance for Images in the Wild**

This is the companion repository for the paper:

> F. Légare and S. I. Sion, "Pixel Block Chain: Spatial Tamper Localization
> and Crop-Resilient Provenance for Images in the Wild," submitted to the
> 4th Workshop on MultiMedia FORensics in the WILD (MMForWILD), IEEE ICIP
> 2026.

## What is PBC?

PBC is a lightweight, **learning-free** watermarking scheme that embeds a grid
of independent hash-linked chains directly into image least-significant bits.
This enables:

- **Spatial tamper localization** — a break in one tile does not cascade to
  adjacent regions (proved formally in the paper).
- **Partial provenance recovery** after cropping and redistribution.
- **Append-mode editing** — compliant editors record per-tile edits in an
  auditable Edit Ledger without re-encoding clean regions.
- **PBC-Forest** — independent genesis blocks at pseudo-random positions for
  crop-heavy distribution paths.

PBC complements manifest-based systems (e.g., C2PA): it provides a pixel-level
residual when the external manifest is stripped by transcoding or platform
re-encoding.

## Headline Results (from the paper)

On a 99-image benchmark spanning real photographs and synthetic content:

| Metric                              | Value                           |
|-------------------------------------|---------------------------------|
| Mean PSNR at embedding depth k=1    | 51.21 ± 0.17 dB                 |
| Tile-level tamper detection         | 100 %                           |
| Tile-level false positive rate      | 0 %                             |
| PBC-Forest fragment survival        | 41.1 – 44.6 % after severe crop |
| Grid-baseline survival (same crop)  | 0 %                             |

## Repository Status

This repository is in **initial release** and currently contains the paper
source so reviewers can reproduce figures and cross-reference equations:

```
paper/
  PBC_MMForWILD2026_v01.tex       LaTeX source of the submitted paper
  figures/                         All figures referenced in the paper
LICENSE                            MIT
README.md                          this file
```

### Forthcoming

The reference implementation, evaluation harness, and benchmark dataset
will be released here in subsequent commits. Planned contents:

- `pbc/` — encoder, verifier, Edit Ledger, PBC-Forest
- `examples/` — scripted reproductions of the paper's experiments
- `data/` — the 99-image benchmark (or a download script if licensing
  precludes redistribution)
- `output/` — reference result logs

Reviewers needing the implementation ahead of the public release should
contact the corresponding author.

## Citation

```bibtex
@inproceedings{legare2026pbc,
  title     = {Pixel Block Chain: Spatial Tamper Localization and
               Crop-Resilient Provenance for Images in the Wild},
  author    = {L{\'e}gare, Fran{\c{c}}ois and Sion, Sion Israel},
  booktitle = {Proc. 4th Workshop on MultiMedia FORensics in the WILD
               (MMForWILD), IEEE ICIP},
  year      = {2026}
}
```

## Contact

- François Légare — flegare@gmail.com (corresponding author)
- Sion Israel Sion — École de technologie Supérieure de Montréal

## License

Released under the MIT License — see [LICENSE](LICENSE).
