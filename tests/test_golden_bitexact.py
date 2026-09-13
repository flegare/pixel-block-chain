"""
Golden bit-exact regression tests for the PBC format.

tests/golden/pbc_golden.json was generated with the original pure-Python
reference implementation, before any optimisation. Every encoder output must
stay byte-identical and every verifier result field-identical (tile and block
statuses, offsets, parsed block fields, Edit Ledger entries), so images that
are already protected keep verifying and the paper's format contract holds.

Regenerate ONLY for an intentional, versioned format change:
    python tests/test_golden_bitexact.py --regenerate
"""

import dataclasses
import functools
import hashlib
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pbc import OpCode, compute_grid  # noqa: E402
from pbc.encoder import encode, encode_region, append_edit, encode_sequence  # noqa: E402
from pbc.decoder import verify, extract_edit_ledger  # noqa: E402
from pbc.scatter import (scatter_encode, scatter_verify,  # noqa: E402
                         scatter_forest_encode, scatter_forest_verify)
from pbc.video import encode_video, verify_video  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "pbc_golden.json")
LEO = os.path.join(ROOT, "examples", "img", "leo.jpg")
TS = 1_700_000_000
CAMERA, EDITOR = "golden-camera", "golden-editor"


# ---------------------------------------------------------------- hashing ----

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def arr_sha(a: np.ndarray) -> str:
    return _sha(f"{a.shape}|{a.dtype}|".encode() + np.ascontiguousarray(a).tobytes())


def obj_sha(obj) -> str:
    return _sha(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def block_repr(b):
    if b is None:
        return None
    return [b.sync.hex(), b.version, b.originator_id, int(b.opcode), b.block_index,
            b.tile_x, b.tile_y, b.timestamp_delta, b.extension, b.crc16, b.chain_hash.hex()]


def grid_repr(r):
    return {"dims": [r.width, r.height, r.cols, r.rows], "overall": int(r.overall_status),
            "tiles": [[t.tx, t.ty, int(t.status), t.block_count, t.originator_id,
                       [[int(b.status), b.block_index, b.pixel_start, b.pixel_end,
                         int(b.opcode), b.originator_id, block_repr(b.block)] for b in t.blocks],
                       [[e.originator_id, int(e.opcode), e.timestamp_delta, e.start_block, e.end_block]
                        for e in extract_edit_ledger(t)]]
                      for row in r.tile_results for t in row]}


def verified(img, **kw):
    return {"verify": obj_sha(grid_repr(verify(img, **kw))),
            "verify_strict": obj_sha(grid_repr(verify(img, strict=True, **kw)))}


# ----------------------------------------------------------------- inputs ----

def _random(w, h, seed):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _gradient(w, h):
    y, x = np.mgrid[0:h, 0:w]
    return np.stack([x * 255 // (w - 1), y * 255 // (h - 1),
                     (x + y) * 255 // (w + h - 2)], axis=-1).astype(np.uint8)


def _leo():
    return np.array(Image.open(LEO).convert("RGB"), dtype=np.uint8)   # native 978×678


INPUTS = {"rand512": lambda: _random(512, 512, 1),
          "rand333x257": lambda: _random(333, 257, 2),
          "leo978x678": _leo,
          "grad1024x768": lambda: _gradient(1024, 768)}


@functools.lru_cache(maxsize=None)
def source(key):
    return INPUTS[key]()


@functools.lru_cache(maxsize=None)
def encoded(key):
    return encode(source(key), originator=CAMERA, timestamp=TS)


def _mask(img):
    H, W = img.shape[:2]
    cols, rows, tw, th = compute_grid(W, H)
    m = np.zeros((H, W), dtype=bool)
    m[th // 2: th // 2 + th, tw // 2: tw // 2 + 10] = True       # spans two tile rows
    return m


def _tamper(key, kind):
    enc = encoded(key)
    H, W = enc.shape[:2]
    cols, rows, tw, th = compute_grid(W, H)
    a = enc.copy()
    if kind == "rect_random":
        a[th // 4: th // 4 + 40, tw // 3: tw // 3 + 60] = \
            np.random.default_rng(7).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    elif kind == "band_constant":                                 # wipes whole blocks
        a[th + 10: th + 30, :] = (255, 0, 255)
    elif kind == "tile_copy":                                     # exact same-size tile copy
        a[th: 2 * th, tw: 2 * tw] = enc[0: th, 0: tw]
    elif kind == "sync_bitflips":                                 # 3 sync bits of block 5, tile (0,0)
        p = 5 * 86
        for off, ch in ((0, 0), (2, 1), (4, 2)):
            a[(p + off) // tw, (p + off) % tw, ch] ^= 1
    return a


# ------------------------------------------------------------------ cases ----

def _case_encode(key):
    return {"input": arr_sha(source(key)), "encoded": arr_sha(encoded(key)), **verified(encoded(key))}


def _case_tamper(key, kind):
    return {"input": arr_sha(source(key)), **verified(_tamper(key, kind))}


def _case_append(key):
    out = append_edit(encoded(key), EDITOR, OpCode.EDIT_COLOR, timestamp=TS + 3600,
                      region_mask=_mask(encoded(key)), split_fraction=0.33)
    return {"input": arr_sha(source(key)), "encoded": arr_sha(out), **verified(out)}


def _case_region(key):
    out = encode_region(encoded(key), _mask(encoded(key)), EDITOR, OpCode.EDIT_RETOUCH,
                        timestamp=TS + 7200)
    return {"input": arr_sha(source(key)), "encoded": arr_sha(out), **verified(out)}


def _case_small_extras():
    img = source("rand333x257")
    out = {}
    for k in (2, 3, 4):
        e = encode(img, originator=CAMERA, timestamp=TS, k=k)
        out[f"k{k}_encoded"] = arr_sha(e)
        out[f"k{k}_verify"] = obj_sha(grid_repr(verify(e, k=k)))
    seq = encode_sequence(img, [(CAMERA, OpCode.CAMERA_ISP, 3, 0),
                                (EDITOR, OpCode.BATCH_TONAL, 1, (5 << 16) | 0x13),
                                ("golden-ai", OpCode.EDIT_AI_GENERATE, 40, 0)], timestamp=TS)
    out["sequence_encoded"] = arr_sha(seq)
    out["sequence_verify"] = obj_sha(grid_repr(verify(seq)))
    out["append_nomask_encoded"] = arr_sha(append_edit(encoded("rand333x257"), EDITOR,
                                                       OpCode.EDIT_CROP, timestamp=TS + 1))
    out["unencoded_verify"] = obj_sha(grid_repr(verify(img)))
    tiny = _random(8, 8, 16)
    out["tiny_encoded"] = arr_sha(encode(tiny, originator=CAMERA, timestamp=TS))
    out["tiny_verify"] = obj_sha(grid_repr(verify(tiny)))
    return {"input": arr_sha(img), **out}


def _case_scatter():
    img = source("rand333x257")
    sc = scatter_encode(img, CAMERA, n_blocks=150, seed=3, timestamp=TS)
    sr = scatter_verify(sc)
    fo = scatter_forest_encode(img, CAMERA, n_blocks=200, seed=5, timestamp=TS)
    fr = scatter_forest_verify(fo[10:200, 20:300])                # a non-aligned crop
    return {"input": arr_sha(img), "scatter_encoded": arr_sha(sc), "forest_encoded": arr_sha(fo),
            "scatter_verify": obj_sha([sr.width, sr.height, sr.n_chains, sr.n_candidates,
                                       [dataclasses.asdict(c) for c in sr.chains]]),
            "forest_verify": obj_sha([fr.width, fr.height, fr.n_genesis_found, fr.n_candidates])}


def _case_video():
    base = source("rand333x257")
    frames = [np.roll(base, 7 * i, axis=1) for i in range(3)]
    enc = encode_video(frames, CAMERA, timestamp=TS)
    swapped = [enc[0], enc[2], enc[1]]
    res = []
    for seq in (enc, swapped):
        for r in verify_video(seq):
            detail = {f"{tx},{ty}": {k: (v.hex() if isinstance(v, bytes) else v) for k, v in d.items()}
                      for (tx, ty), d in r.inter_detail.items()}
            res.append([r.frame_index, r.inter_ok, detail, grid_repr(r.intra_result)])
    return {"input": arr_sha(base), "frames_encoded": [arr_sha(e) for e in enc], "verify": obj_sha(res)}


CASES = {}
for _key in INPUTS:
    CASES[f"{_key}/encode"] = functools.partial(_case_encode, _key)
    for _kind in ("rect_random", "band_constant", "tile_copy", "sync_bitflips"):
        CASES[f"{_key}/tamper_{_kind}"] = functools.partial(_case_tamper, _key, _kind)
    CASES[f"{_key}/append_edit"] = functools.partial(_case_append, _key)
    CASES[f"{_key}/encode_region"] = functools.partial(_case_region, _key)
CASES["rand333x257/k_sequence_extras"] = _case_small_extras
CASES["rand333x257/scatter_forest"] = _case_scatter
CASES["rand333x257/video"] = _case_video


# ------------------------------------------------------------------ tests ----

def _golden():
    with open(GOLDEN) as f:
        return json.load(f)


@pytest.mark.parametrize("name", sorted(CASES))
def test_bitexact_against_reference(name):
    expected = _golden()[name]
    actual = CASES[name]()
    if actual["input"] != expected["input"]:
        pytest.skip(f"input decoded differently on this machine ({name}); "
                    f"cannot compare (likely a different JPEG decoder)")
    assert actual == expected


if __name__ == "__main__":
    if "--regenerate" not in sys.argv:
        sys.exit("refusing to overwrite golden fixtures without --regenerate")
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    data = {}
    for name in sorted(CASES):
        data[name] = CASES[name]()
        print(f"  {name}")
    with open(GOLDEN, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    print(f"wrote {len(data)} golden cases to {GOLDEN}")
