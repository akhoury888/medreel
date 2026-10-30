"""Procedural sound design (original, royalty-free by construction): whooshes on scene changes, soft pops when
cards and keywords appear, a low 'impact' under the opening title. Mixed under the voice at a modest level."""
import wave
import numpy as np

SR = 44100
_RNG = np.random.default_rng(3)


def _onepole_lp(x, fc):
    """Time-varying one-pole low-pass (fc: array of cut-offs in Hz, same length as x)."""
    a = np.exp(-2 * np.pi * np.asarray(fc) / SR)
    y = np.empty_like(x); acc = 0.0
    for i in range(len(x)):
        acc = (1 - a[i]) * x[i] + a[i] * acc; y[i] = acc
    return y


def whoosh(dur=0.5, up=True):
    n = int(dur * SR); t = np.linspace(0, 1, n)
    noise = _RNG.normal(0, 1, n)
    env = np.sin(np.pi * np.clip(t, 0, 1)) ** 1.6 * (t ** 0.4 if up else (1 - t) ** 0.4 + 0.2)
    fc = 300 + 5200 * np.sin(np.pi * t) ** 2  # filter opens then closes: air rushing past
    lp = _onepole_lp(noise, fc); hp = lp - _onepole_lp(lp, np.full(n, 180.0))
    out = hp * env
    return out / (np.abs(out).max() + 1e-9)


def pop(dur=0.14, f0=950.0, f1=520.0):
    n = int(dur * SR); t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / 0.025)
    ph = 2 * np.pi * np.cumsum(f) / SR
    out = np.sin(ph) * np.exp(-t / 0.035) + 0.25 * np.sin(2 * ph) * np.exp(-t / 0.02)
    out[: int(0.002 * SR)] *= np.linspace(0, 1, int(0.002 * SR))
    return out / (np.abs(out).max() + 1e-9)


def impact(dur=1.1):
    n = int(dur * SR); t = np.arange(n) / SR
    f = 46 + 60 * np.exp(-t / 0.08)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.35)
    burst = _onepole_lp(_RNG.normal(0, 1, n), np.full(n, 900.0)) * np.exp(-t / 0.05) * 0.8
    out = body + burst
    return out / (np.abs(out).max() + 1e-9)


def shimmer(dur=0.6):
    n = int(dur * SR); t = np.arange(n) / SR
    out = sum(np.sin(2 * np.pi * f * t + k) * np.exp(-t / (0.18 + 0.05 * k)) for k, f in enumerate((1568, 2093, 2637, 3136)))
    out *= np.minimum(1, t / 0.01)
    return out / (np.abs(out).max() + 1e-9)


SOUNDS = {"whoosh": (whoosh, 0.55), "whoosh_out": (lambda: whoosh(0.4, up=False), 0.35), "pop": (pop, 0.45),
          "impact": (impact, 0.7), "shimmer": (shimmer, 0.25)}  # name -> (generator, gain)


def render_track(events, total, path, level_db=-6.0):
    """events: [(time_s, name)] -> stereo 16-bit WAV of length `total` (silence elsewhere)."""
    n = int((total + 1.5) * SR); buf = np.zeros(n, np.float32)
    cache = {}
    for t, name in events:
        if name not in SOUNDS or t < 0 or t >= total: continue
        if name not in cache: cache[name] = SOUNDS[name][0]().astype(np.float32) * SOUNDS[name][1]
        x = cache[name]; i = int(t * SR); j = min(n, i + len(x))
        buf[i:j] += x[: j - i]
    peak = np.abs(buf).max()
    if peak > 1: buf /= peak
    buf *= 10 ** (level_db / 20)
    pcm = (np.clip(buf, -1, 1) * 32767).astype(np.int16)
    st = np.repeat(pcm[:, None], 2, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(st.tobytes())
    return path
