"""Procedurally generated background music beds (original, royalty-free by construction).
Replace/extend with a licensed library (e.g. Epidemic/Artlist business plans) in production."""
import os, wave
import numpy as np

SR = 44100
CHORDS = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]  # Am F C G (MIDI)
PROG = {"calm": [(60, 64, 67), (55, 59, 62), (57, 60, 64), (53, 57, 60)],       # C G Am F — warm, major
        "pulse": CHORDS,                                                        # Am F C G — modern explainer
        "drive": [(64, 67, 71), (60, 64, 67), (55, 59, 62), (62, 66, 69)],      # Em C G D — bright, energetic
        "cinematic": [(50, 53, 57), (46, 50, 53), (53, 57, 60), (48, 52, 55)]}  # Dm Bb F C — dark, wide
TRACKS = {"calm": dict(bpm=72, kick=0, hat=0, bass=0, arp=0.5, pad=1.0, drone=0),
          "pulse": dict(bpm=100, kick=0.5, hat=0.18, bass=0.35, arp=0.45, pad=0.8, drone=0),
          "drive": dict(bpm=118, kick=0.85, hat=0.28, bass=0.6, arp=0.4, pad=0.6, drone=0, clap=0.35),
          "cinematic": dict(bpm=60, kick=0, hat=0, bass=0, arp=0.2, pad=0.9, drone=0.7, boom=0.6)}


def _hz(m): return 440 * 2 ** ((m - 69) / 12)


def _lowpass(x, fc):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (f / fc) ** 4)
    return np.fft.irfft(X, len(x))


def generate(track, seconds, out_path):
    p = TRACKS.get(track, TRACKS["calm"])
    n = int(seconds * SR); t = np.arange(n) / SR
    beat = 60 / p["bpm"]; bar = beat * 4
    rng = np.random.default_rng(1)
    pad = np.zeros(n); bass = np.zeros(n); arp = np.zeros(n); perc = np.zeros(n)
    nb = int(np.ceil(seconds / bar)) + 1
    for b in range(nb):
        ch = PROG.get(track, CHORDS)[b % 4]; s0 = int(b * bar * SR); s1 = min(n, int((b + 1) * bar * SR) + int(0.3 * SR))
        if s0 >= n: break
        tt = t[s0:s1] - t[s0]
        env = np.minimum(1, tt / 0.6) * np.exp(-np.maximum(0, tt - bar) * 4)
        for m in ch:
            for det in (-0.08, 0.08):
                ph = 2 * np.pi * _hz(m + det) * tt
                pad[s0:s1] += env * (0.5 * np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph))
        if p["bass"]:
            for k in range(8):
                q0 = s0 + int(k * beat / 2 * SR); q1 = min(n, q0 + int(beat / 2 * SR))
                if q0 >= n: break
                tq = t[q0:q1] - t[q0]
                bass[q0:q1] += np.sin(2 * np.pi * _hz(ch[0] - 24) * tq) * np.exp(-tq * 6)
        if p["arp"]:
            seq = [ch[0] + 12, ch[1] + 12, ch[2] + 12, ch[1] + 24]
            for k in range(8):
                q0 = s0 + int(k * beat / 2 * SR); q1 = min(n, q0 + int(0.5 * SR))
                if q0 >= n: break
                tq = t[q0:q1] - t[q0]
                arp[q0:q1] += np.sin(2 * np.pi * _hz(seq[k % 4]) * tq) * np.exp(-tq * 7) * (0.7 + 0.3 * np.sin(2 * np.pi * 3 * tq))
        for k in range(4):
            q0 = s0 + int(k * beat * SR)
            if q0 >= n: break
            if p["kick"]:
                q1 = min(n, q0 + int(0.35 * SR)); tq = t[q0:q1] - t[q0]
                if p["bpm"] < 110 and k % 2: pass
                else: perc[q0:q1] += p["kick"] * np.sin(2 * np.pi * (50 + 90 * np.exp(-tq * 30)) * tq) * np.exp(-tq * 9)
            if p.get("clap") and k % 2:
                q1 = min(n, q0 + int(0.2 * SR)); tq = t[q0:q1] - t[q0]
                perc[q0:q1] += p["clap"] * rng.normal(0, 1, q1 - q0) * np.exp(-tq * 25) * 0.5
            if p["hat"]:
                for hh in (0, 0.5):
                    h0 = q0 + int(hh * beat * SR); h1 = min(n, h0 + int(0.06 * SR))
                    if h0 < n: perc[h0:h1] += p["hat"] * np.diff(rng.normal(0, 1, h1 - h0 + 1)) * np.exp(-(t[h0:h1] - t[h0]) * 60) * 0.4
            if p.get("boom") and k == 0:
                q1 = min(n, q0 + int(2 * SR)); tq = t[q0:q1] - t[q0]
                perc[q0:q1] += p["boom"] * np.sin(2 * np.pi * 42 * tq) * np.exp(-tq * 1.6)
    mix = 0.22 * _lowpass(pad, 1800) * p["pad"] + 0.35 * _lowpass(bass, 400) * p["bass"] + 0.16 * arp * p["arp"] + 0.4 * perc
    if p["drone"]:
        mix += p["drone"] * 0.12 * _lowpass(np.sin(2 * np.pi * 55 * t) + 0.5 * np.sin(2 * np.pi * 82.4 * t) + 0.3 * rng.normal(0, 1, n), 300) * (0.6 + 0.4 * np.sin(2 * np.pi * t / 8))
    mix *= np.minimum(1, t / 1.5) * np.minimum(1, (seconds - t) / 2.0)
    mix /= np.abs(mix).max() + 1e-9
    L = mix; R = np.roll(mix, int(0.012 * SR)) * 0.95
    st = (np.stack([L, R], 1) * 0.85 * 32767).astype(np.int16)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(st.tobytes())
    return out_path
