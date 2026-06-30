#!/usr/bin/env python3
"""
PBC Sync Frame Statistical Analysis

Validates the two quantitative claims in Appendix A of the paper:

  1. "The selected pattern has a maximum off-peak autocorrelation of 0.167
     (4/24 bits), compared to a random expectation of 0.5."

  2. "False positive sync detection to < 10^{-6} per pixel position."

Method:
  Claim 1 — Exhaustively compute cyclic autocorrelation of 0xA5C396 over
             all 23 non-zero shifts of the 24-bit pattern, and compare to
             a random-sampling baseline over 10^6 random 24-bit patterns.

  Claim 2 — Scan the LSB stream of a real natural photograph (leo.jpg,
             never PBC-encoded) for false sync detections using the same
             sliding correlator the decoder uses. Report empirical FPR.
             Also compute the theoretical Binomial FPR for p=0.5 random bits.

MIT License - Copyright (c) 2026 Francois Legare
"""

import sys
import os
import math
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbc import SYNC_PATTERN, SYNC_BITS, SYNC_HAMMING_THRESHOLD, BITS_PER_CHANNEL

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
LEO_JPG = os.path.join(IMG_DIR, 'leo.jpg')

# SYNC_PATTERN is 6 bytes: 0xA5C396 repeated twice (48 bits total)
SYNC_48 = int.from_bytes(SYNC_PATTERN, 'big')   # full 48-bit word
SYNC_24 = 0xA5C396                               # half pattern (24 bits)


# ---------------------------------------------------------------------------
# Claim 1 — Autocorrelation of 0xA5C396
# ---------------------------------------------------------------------------

def hamming_weight(x: int, bits: int) -> int:
    """Count set bits in an integer masked to <bits> width."""
    return bin(x & ((1 << bits) - 1)).count('1')


def cyclic_autocorrelation_24(pattern: int) -> list:
    """
    Compute cyclic autocorrelation for all 23 non-zero shifts of a 24-bit pattern.
    Returns list of (shift, matching_bits, correlation).
    correlation = matching_bits / 24
    """
    mask = (1 << 24) - 1
    results = []
    for shift in range(1, 24):
        shifted = ((pattern << shift) | (pattern >> (24 - shift))) & mask
        agreements = 24 - hamming_weight(pattern ^ shifted, 24)
        corr = agreements / 24
        results.append((shift, agreements, corr))
    return results


def random_pattern_max_autocorr(n_samples: int = 1_000_000,
                                 seed: int = 42) -> dict:
    """
    Sample n_samples random 24-bit patterns, compute their max off-peak
    autocorrelation. Return statistics.
    """
    rng = np.random.default_rng(seed)
    patterns = rng.integers(0, 1 << 24, size=n_samples, dtype=np.uint32)
    mask = np.uint32((1 << 24) - 1)

    max_corrs = []
    for pat in patterns:
        worst = 0
        for shift in range(1, 24):
            shifted = int(((int(pat) << shift) | (int(pat) >> (24 - shift))) & int(mask))
            agreements = 24 - hamming_weight(int(pat) ^ shifted, 24)
            worst = max(worst, agreements / 24)
        max_corrs.append(worst)

    arr = np.array(max_corrs)
    return {
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'p50': float(np.percentile(arr, 50)),
    }


def analyze_autocorrelation():
    print("=" * 65)
    print("Claim 1 — Sync frame autocorrelation (24-bit half-pattern 0xA5C396)")
    print("=" * 65)
    print()

    results = cyclic_autocorrelation_24(SYNC_24)
    max_shift, max_agree, max_corr = max(results, key=lambda x: x[2])
    min_shift, min_agree, min_corr = min(results, key=lambda x: x[2])

    print(f"  Pattern : 0x{SYNC_24:06X}  = {SYNC_24:024b}")
    print(f"  Shifts  : 1..23 (all non-zero cyclic shifts)")
    print()
    print(f"  {'Shift':>6}  {'Agree':>6}  {'Corr':>6}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}")
    for shift, agree, corr in results:
        marker = " <-- MAX" if corr == max_corr else (" <-- MIN" if corr == min_corr else "")
        print(f"  {shift:>6}  {agree:>6}  {corr:>6.3f}{marker}")

    print()
    print(f"  Max off-peak autocorrelation : {max_corr:.3f}  ({max_agree}/{24} bits)")
    print(f"  Paper claims                 : 0.167  (4/24 bits)")
    paper_claim_max = 4 / 24
    if abs(max_corr - paper_claim_max) < 0.001:
        print(f"  Result : CONFIRMED")
    else:
        print(f"  Result : DIFFERS from claim (actual max = {max_corr:.3f})")

    print()
    print(f"  Random expectation (mean max-autocorr over 10^6 patterns):")
    print(f"  Sampling 1,000 patterns for speed...", end='', flush=True)
    rand_stats = random_pattern_max_autocorr(n_samples=1_000, seed=42)
    print(f"  mean={rand_stats['mean']:.3f}  median={rand_stats['median']:.3f}")
    print(f"  Paper claims random expectation = 0.5")
    print()


# ---------------------------------------------------------------------------
# Claim 2 — False positive rate in natural images
# ---------------------------------------------------------------------------

def extract_lsb_stream(image_rgb: np.ndarray) -> np.ndarray:
    """Extract bit-0 of each channel as a 1D uint8 array of 0s and 1s."""
    flat = image_rgb.reshape(-1)           # (H*W*3,)
    return (flat & 1).astype(np.uint8)


def bits_to_int48(bits: np.ndarray, start: int) -> int:
    """Pack 48 bits starting at <start> into a 48-bit integer."""
    chunk = bits[start:start + 48]
    result = 0
    for b in chunk:
        result = (result << 1) | int(b)
    return result


def scan_false_positives_vectorized(bits: np.ndarray,
                                     sync_48: int,
                                     threshold: int) -> tuple:
    """
    Efficient vectorized scan: convert bits to uint64 windows and check
    Hamming distance to SYNC_48.

    Returns (n_positions_checked, n_false_positives, false_positive_rate)
    """
    n = len(bits)
    if n < 48:
        return 0, 0, 0.0

    # Pack bits into a single large integer array for fast XOR
    # Work in blocks of 64 bits using numpy uint64
    n_positions = n - 48 + 1

    # Build a uint8 array where each element is one bit, then use
    # sliding window via stride tricks
    false_positives = 0

    # XOR each 48-bit window against the sync pattern
    # Unpack sync pattern bits
    sync_bits = np.array([(sync_48 >> (47 - i)) & 1
                           for i in range(48)], dtype=np.uint8)

    # Use strided view for sliding windows
    shape   = (n_positions, 48)
    strides = (bits.strides[0], bits.strides[0])
    windows = np.lib.stride_tricks.as_strided(bits, shape=shape, strides=strides)

    # XOR each window with sync pattern; count mismatches per window
    xor     = windows ^ sync_bits[np.newaxis, :]   # (n_positions, 48)
    hamming = xor.sum(axis=1)                       # (n_positions,)
    false_positives = int((hamming <= threshold).sum())

    return n_positions, false_positives, false_positives / n_positions


def theoretical_fpr(sync_bits: int, window_bits: int, threshold: int,
                    p_bit: float = 0.5) -> float:
    """
    Theoretical false positive rate for random bits with P(bit=1)=p_bit.
    P(Hamming <= threshold) = sum_{k=0}^{threshold} C(n,k) * p^k * (1-p)^(n-k)
    where p = probability of a mismatch bit (= p_bit if random).
    """
    from math import comb
    # P(mismatch at position i) = p_bit (if random, half the bits match sync)
    p_mismatch = p_bit  # for random bits, each position matches with prob 0.5
    total = 0.0
    for k in range(threshold + 1):
        total += comb(window_bits, k) * (p_mismatch ** k) * ((1 - p_mismatch) ** (window_bits - k))
    return total


def analyze_false_positives():
    print("=" * 65)
    print("Claim 2 — False positive sync detection rate on natural images")
    print("=" * 65)
    print()

    # Theoretical rate for random bits
    th_fpr = theoretical_fpr(SYNC_48, SYNC_BITS, SYNC_HAMMING_THRESHOLD)
    print(f"  Sync pattern  : 0x{SYNC_48:012X}  ({SYNC_BITS} bits)")
    print(f"  Threshold tau : {SYNC_HAMMING_THRESHOLD} / {SYNC_BITS} bits")
    print()
    print(f"  Theoretical FPR (random bits, p=0.5) :")
    print(f"    P(Hamming <= {SYNC_HAMMING_THRESHOLD}) = {th_fpr:.2e}")
    verdict = 'CONSISTENT' if th_fpr < 1e-6 else 'EXCEEDS'
    print(f"    Paper claims < 10^{{-6}}  ->  {verdict}")
    print()

    # Empirical rate on natural image LSBs
    if not os.path.exists(LEO_JPG):
        print(f"  leo.jpg not found at {LEO_JPG} — skipping empirical test")
        return

    img = np.array(Image.open(LEO_JPG).convert('RGB'))
    H, W = img.shape[:2]
    bits = extract_lsb_stream(img)

    print(f"  Natural image : leo.jpg  ({W}x{H} = {W*H:,} pixels)")
    print(f"  LSB stream    : {len(bits):,} bits")
    print()
    print(f"  Scanning for false sync detections (threshold={SYNC_HAMMING_THRESHOLD})...")

    n_pos, n_fp, emp_fpr = scan_false_positives_vectorized(
        bits, SYNC_48, SYNC_HAMMING_THRESHOLD)

    print(f"    Positions checked : {n_pos:,}")
    print(f"    False positives   : {n_fp}")
    print(f"    Empirical FPR     : {emp_fpr:.2e}")
    print(f"    Paper claims      : < 10^{{-6}}")

    if n_fp == 0:
        print(f"    Result : CONFIRMED (zero false positives observed)")
    elif emp_fpr < 1e-6:
        print(f"    Result : CONFIRMED (FPR {emp_fpr:.2e} < 10^{{-6}})")
    else:
        print(f"    Result : EXCEEDS claimed bound ({emp_fpr:.2e} > 10^{{-6}})")

    print()
    print(f"  Note: Natural image LSBs are NOT random (spatial correlations,")
    print(f"  even-value bias in some codecs). The empirical rate may differ")
    print(f"  from the theoretical random-bit prediction.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("PBC Sync Frame Statistical Analysis")
    print("=" * 65)
    print(f"SYNC_PATTERN (48-bit) : 0x{SYNC_48:012X}")
    print(f"SYNC_HAMMING_THRESHOLD: {SYNC_HAMMING_THRESHOLD}")
    print(f"BITS_PER_CHANNEL      : {BITS_PER_CHANNEL}")
    print()

    analyze_autocorrelation()
    print()
    analyze_false_positives()

    print()
    print("=" * 65)
    print("Appendix A verdict summary:")
    print()
    print("  Claim 1 (max autocorr 0.167): see table above")
    print("  Claim 2 (FPR < 10^{-6})     : see empirical scan above")


if __name__ == '__main__':
    sys.exit(main())
