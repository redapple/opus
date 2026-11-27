#!/usr/bin/env python3
"""Deterministic synthetic test clip: 48 kHz, stereo, 16-bit little-endian raw.

Harmonic tones with vibrato, coloured noise and decaying noise bursts, so the
encoder sees tonal, noisy and transient frames.  Standard library only.
Usage: gen_audio.py OUT.raw SECONDS
"""
import array
import math
import sys

RATE = 48000


def main(path, seconds):
    n = RATE * seconds
    state = 123456789

    def rnd():
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return state / 0x40000000 - 1.0

    out = array.array("h", [0]) * (2 * n)
    lp = [0.0, 0.0]
    burst = 0.0
    two_pi = 2.0 * math.pi
    for i in range(n):
        t = i / RATE
        f0 = 220.0 * (1.0 + 0.01 * math.sin(two_pi * 5.0 * t))
        tone = 0.0
        for h in range(1, 8):
            tone += math.sin(two_pi * f0 * h * t) / h
        if i % (RATE // 2) == 0:
            burst = 1.0
        burst *= 0.9985
        for ch in range(2):
            lp[ch] += 0.15 * (rnd() - lp[ch])
            ph = 0.3 * ch
            s = 0.25 * tone * (1.0 + 0.1 * ch) + 0.30 * lp[ch] + 0.45 * burst * rnd()
            s += 0.1 * math.sin(two_pi * 3000.0 * t + ph) * (0.5 + 0.5 * math.sin(two_pi * 0.7 * t))
            v = int(s * 9000.0)
            out[2 * i + ch] = max(-32768, min(32767, v))
    with open(path, "wb") as f:
        f.write(out.tobytes())


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
