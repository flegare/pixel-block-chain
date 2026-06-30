#!/usr/bin/env python3
"""
PBC Audio (MP3) Robustness Test

Measures bit-error rates per bit-position in 16-bit PCM audio after MP3
round-trips at various bitrates, using ffmpeg for encoding/decoding.

No audio PBC implementation exists yet; this script tests the raw channel
assumption: "MP3 poses the same fundamental challenge as JPEG for LSBs."
We compare bit-error profiles for both to characterize whether the analogy
holds, and whether MP3 is better, worse, or differently destructive.

Results feed directly into Section 9.5 of the paper.

MIT License - Copyright (c) 2026 Francois Legare
"""

import sys
import os
import io
import struct
import wave
import subprocess
import tempfile
from pathlib import Path
import numpy as np

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')

# MP3 bitrates to test (kbps), highest to lowest
BITRATES = [320, 256, 192, 128, 96, 64]

# Audio parameters
SAMPLE_RATE   = 44100
CHANNELS      = 2
DURATION_SECS = 5      # enough samples for reliable statistics
N_SAMPLES     = SAMPLE_RATE * CHANNELS * DURATION_SECS  # per-channel interleaved

# Sync-frame analogy: how many bit-positions to check
BIT_POSITIONS = [0, 1, 2, 3, 4, 5]

# JPEG reference values from jpeg_robustness empirical test (for comparison)
JPEG_Q100_BERS = {0: 0.420, 1: 0.190, 2: 0.095, 3: 0.046}


# ---------------------------------------------------------------------------
# WAV I/O helpers (stdlib only)
# ---------------------------------------------------------------------------

def make_wav_bytes(samples_i16: np.ndarray, sample_rate: int, n_channels: int) -> bytes:
    """Encode int16 numpy array as WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples_i16.astype('<i2').tobytes())
    return buf.getvalue()


def read_wav_bytes(data: bytes) -> np.ndarray:
    """Decode WAV bytes to int16 numpy array."""
    buf = io.BytesIO(data)
    with wave.open(buf, 'rb') as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype='<i2').astype(np.int32)


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def wav_to_mp3(wav_bytes: bytes, bitrate_kbps: int) -> bytes:
    """Encode WAV bytes to MP3 bytes via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_in:
        f_in.write(wav_bytes)
        wav_path = f_in.name
    mp3_path = wav_path.replace('.wav', '.mp3')
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', wav_path,
             '-b:a', f'{bitrate_kbps}k',
             '-codec:a', 'libmp3lame',
             mp3_path],
            capture_output=True, check=True
        )
        with open(mp3_path, 'rb') as f:
            return f.read()
    finally:
        os.unlink(wav_path)
        if os.path.exists(mp3_path):
            os.unlink(mp3_path)


def mp3_to_wav(mp3_bytes: bytes) -> bytes:
    """Decode MP3 bytes back to WAV bytes via ffmpeg (s16le PCM)."""
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f_in:
        f_in.write(mp3_bytes)
        mp3_path = f_in.name
    wav_path = mp3_path.replace('.mp3', '_decoded.wav')
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', mp3_path,
             '-acodec', 'pcm_s16le',
             wav_path],
            capture_output=True, check=True
        )
        with open(wav_path, 'rb') as f:
            return f.read()
    finally:
        os.unlink(mp3_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)


# ---------------------------------------------------------------------------
# Bit-error analysis
# ---------------------------------------------------------------------------

def bit_error_rates(original: np.ndarray, decoded: np.ndarray,
                    bit_positions: list) -> dict:
    """
    Compute per-bit-position error rates.

    Both arrays are int32 representations of int16 samples.
    Truncate to the shorter length (MP3 decoder may add/drop a few frames).
    """
    n = min(len(original), len(decoded))
    orig = original[:n].astype(np.int32)
    dec  = decoded[:n].astype(np.int32)

    # Work in unsigned 16-bit space for bit operations
    orig_u = orig.astype(np.uint16)
    dec_u  = dec.astype(np.uint16)
    diff   = orig_u ^ dec_u          # XOR: 1 where bits differ

    bers = {}
    for bit in bit_positions:
        mask = np.uint16(1 << bit)
        flipped = (diff & mask).astype(bool)
        bers[bit] = float(flipped.mean())
    return bers


def sample_error_stats(original: np.ndarray, decoded: np.ndarray) -> dict:
    """Mean and max absolute sample error after round-trip."""
    n = min(len(original), len(decoded))
    err = np.abs(original[:n].astype(np.int32) - decoded[:n].astype(np.int32))
    return {
        'mean': float(err.mean()),
        'median': float(np.median(err)),
        'p95': float(np.percentile(err, 95)),
        'max': int(err.max()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("PBC Audio (MP3) Robustness Test")
    print("=" * 75)
    print(f"Signal : {DURATION_SECS}s, {SAMPLE_RATE} Hz stereo 16-bit PCM, random content")
    print(f"Purpose: Measure bit-error rates per bit-position after MP3 round-trip")
    print(f"         to validate/refute Section 9.5 JPEG analogy claim")
    print()

    # Generate a reproducible random audio signal
    rng = np.random.default_rng(42)
    # Use mid-range values to simulate realistic audio (not clipping)
    samples = rng.integers(-16000, 16000, N_SAMPLES, dtype=np.int16)

    wav_bytes = make_wav_bytes(samples, SAMPLE_RATE, CHANNELS)
    print(f"WAV size: {len(wav_bytes):,} bytes  ({N_SAMPLES:,} samples)")
    print()

    # Header
    col = "{:>8} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>8} | {:>7} | {}"
    print(col.format(
        "Bitrate", "bit0 %", "bit1 %", "bit2 %", "bit3 %", "bit4 %",
        "mean_err", "max_err", "Assessment"))
    print("-" * 95)

    results = []

    for br in BITRATES:
        try:
            mp3_bytes = wav_to_mp3(wav_bytes, br)
            wav_back  = mp3_to_wav(mp3_bytes)
            decoded   = read_wav_bytes(wav_back)
        except subprocess.CalledProcessError as e:
            print(f"{br:>5} kbps | ffmpeg error: {e.stderr.decode()[:60]}")
            continue

        bers  = bit_error_rates(samples.astype(np.int32), decoded, BIT_POSITIONS)
        stats = sample_error_stats(samples.astype(np.int32), decoded)

        # Assessment: can any bit position carry PBC sync reliably?
        # PBC sync threshold is 6/48 bits. For a 48-bit frame:
        # Expected Hamming = 48 * BER. Pass if < 6.
        def expected_hamming(ber): return 48 * ber
        best_bit   = min(BIT_POSITIONS[:4], key=lambda b: bers[b])
        best_hamm  = expected_hamming(bers[best_bit])
        if best_hamm < 6:
            assessment = f"bit{best_bit} sync PASS (H={best_hamm:.1f})"
        else:
            assessment = f"All bits fail (best H={best_hamm:.1f} at bit{best_bit})"

        results.append((br, bers, stats, assessment))

        print(col.format(
            f"{br} kbps",
            f"{bers[0]*100:.1f}%",
            f"{bers[1]*100:.1f}%",
            f"{bers[2]*100:.1f}%",
            f"{bers[3]*100:.1f}%",
            f"{bers[4]*100:.1f}%",
            f"{stats['mean']:.1f}",
            stats['max'],
            assessment))

    # ------------------------------------------------------------------
    # Comparison table vs JPEG
    # ------------------------------------------------------------------
    print()
    print("=" * 75)
    print("Comparison: MP3 320 kbps vs JPEG Q=100 (bit-error rates by position)")
    print()
    if results:
        mp3_best = results[0][1]   # 320 kbps
        print(f"  {'Bit':>4} | {'MP3 320k':>10} | {'JPEG Q100':>10} | {'Worse?':>8}")
        print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
        for bit in range(4):
            mp3_ber  = mp3_best.get(bit, float('nan'))
            jpeg_ber = JPEG_Q100_BERS.get(bit, float('nan'))
            worse    = "MP3" if mp3_ber > jpeg_ber else "JPEG"
            print(f"  bit{bit}  | {mp3_ber*100:>8.1f}%  | {jpeg_ber*100:>8.1f}%  | {worse:>8}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 75)
    print("Summary for paper Section 9.5:")
    print()
    for br, bers, stats, assessment in results:
        print(f"  {br:>3} kbps: bit0={bers[0]*100:.1f}%  bit1={bers[1]*100:.1f}%"
              f"  bit2={bers[2]*100:.1f}%  bit3={bers[3]*100:.1f}%"
              f"  mean_err={stats['mean']:.1f}  -> {assessment}")

    print()
    print("Notes:")
    print("  JPEG Q=100 reference (from jpeg_robustness.py):")
    print("    bit0=42%  bit1=19%  bit2=9.5%  bit3=4.6%")
    print("    Root cause: YCbCr rounding (±1-2 per channel)")
    print()
    print("  MP3 root cause: MDCT transform-domain quantization introduces")
    print("    errors potentially ±tens to ±thousands of sample values,")
    print("    not bounded to ±1 like JPEG YCbCr rounding.")
    print("    Error magnitude determines BER differently from JPEG.")

    # ------------------------------------------------------------------
    # Save results to file
    # ------------------------------------------------------------------
    out_path = Path(__file__).parent.parent / "output" / "results" / "mp3_robustness_results.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("PBC Audio (MP3) Robustness Test Results\n")
        f.write(f"Signal: {DURATION_SECS}s, {SAMPLE_RATE} Hz stereo 16-bit PCM\n\n")
        f.write(f"{'Bitrate':>8} | {'bit0 BER':>8} | {'bit1 BER':>8} | "
                f"{'bit2 BER':>8} | {'bit3 BER':>8} | {'mean_err':>8} | Assessment\n")
        f.write("-" * 85 + "\n")
        for br, bers, stats, assessment in results:
            f.write(f"{br:>5} kbps | {bers[0]*100:>7.1f}% | {bers[1]*100:>7.1f}% | "
                    f"{bers[2]*100:>7.1f}% | {bers[3]*100:>7.1f}% | "
                    f"{stats['mean']:>7.1f}  | {assessment}\n")
        f.write("\nJPEG Q=100 reference: bit0=42%  bit1=19%  bit2=9.5%  bit3=4.6%\n")
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    sys.exit(main())
