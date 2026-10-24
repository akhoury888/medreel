"""Align a doctor's own script to their recording (modes: own script, teleprompter).

The script is the source of truth for caption wording — this sidesteps dialect ASR errors.
ASR is used only for timing anchors: script tokens are matched to ASR words on a fuzzy
"skeleton" key, anchored tokens take the ASR word's time, and the tokens between two anchors
are spread over the speech (energy-VAD) regions in that window by character mass."""
import difflib, re
import numpy as np
from . import knowledge as K

AR_WEAK = re.compile(r"[اويىهةء]")


def detect_script_language(text):
    if K.is_ar(text): return "ar"
    t = " " + text.lower() + " "
    fr = len(re.findall(r"[éèêàùçôîœ]", t)) + 2 * len(re.findall(r" (le|la|les|des|une|est|et|pour|vous|votre|qui|dans|avec|c'est) ", t))
    en = 2 * len(re.findall(r" (the|and|is|you|your|of|to|with|this|that|it's) ", t))
    return "fr" if fr > en else "en"


def _key(w):
    c = K.core(w)
    if K.is_ar(c): return AR_WEAK.sub("", c)[:4] or c[:2]
    return c[:5]


def _spread(tokens, t0, t1, regions):
    """Distribute tokens over speech regions inside [t0, t1] by character mass."""
    regs = [(max(a, t0), min(b, t1)) for a, b in regions if b > t0 and a < t1]
    regs = [r for r in regs if r[1] - r[0] > 0.05] or [(t0, max(t1, t0 + 0.3 * len(tokens)))]
    wts = [len(K.core(t)) + 1 for t in tokens]; W = float(sum(wts))
    D = np.array([b - a for a, b in regs]); bounds = np.concatenate([[0], np.cumsum(D)]) / D.sum()
    out, acc = [], 0.0
    for tok, wt in zip(tokens, wts):
        f0, f1 = acc / W, (acc + wt) / W; acc += wt
        def at(f):
            k = int(min(len(regs) - 1, np.searchsorted(bounds, f, side="right") - 1))
            a, b = regs[k]; return a + (f - bounds[k]) / max(1e-9, bounds[k + 1] - bounds[k]) * (b - a)
        s, e = at(f0), at(min(f1, 1.0) - 1e-6)
        out.append((round(s, 3), round(max(e, s + 0.05), 3)))
    return out


def _skel(w):
    c = K.core(w)
    return AR_WEAK.sub("", c) or c if K.is_ar(c) else re.sub(r"[aeiouy']", "", c) or c


def _sim(a, b):
    if not a or not b: return 0.0
    if a == b: return 1.0
    return 1.0 - K._lev(a, b) / max(len(a), len(b))


def _nw_anchors(tokens, asr, band=0.25):
    """Banded Needleman-Wunsch on consonant skeletons (tolerant to dialect / ASR spelling drift).
    Returns monotonic (script_idx, asr_idx) pairs whose similarity is high enough to trust."""
    n, m = len(tokens), len(asr)
    if not n or not m: return []
    A = [_skel(t) for t in tokens]; B = [_skel(t) for t in asr]
    GAP = -0.45; NEG = -1e9
    S = [[NEG] * (m + 1) for _ in range(n + 1)]; T = [[0] * (m + 1) for _ in range(n + 1)]
    S[0][0] = 0.0
    for j in range(1, m + 1): S[0][j] = S[0][j - 1] + GAP; T[0][j] = 2
    for i in range(1, n + 1):
        S[i][0] = S[i - 1][0] + GAP; T[i][0] = 1
        c = i / n * m; lo, hi = max(1, int(c - band * m) - 3), min(m, int(c + band * m) + 3)
        for j in range(lo, hi + 1):
            sim = _sim(A[i - 1], B[j - 1])
            d = S[i - 1][j - 1] + (2.0 * sim - 0.6 if sim >= 0.45 else -0.8)
            u = S[i - 1][j] + GAP; l = S[i][j - 1] + GAP
            if d >= u and d >= l: S[i][j], T[i][j] = d, 0
            elif u >= l: S[i][j], T[i][j] = u, 1
            else: S[i][j], T[i][j] = l, 2
    i, j, out = n, m, []
    while i > 0 and j > 0:
        t = T[i][j]
        if t == 0:
            if _sim(A[i - 1], B[j - 1]) >= (0.55 if len(A[i - 1]) >= 3 else 0.99): out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif t == 1: i -= 1
        else: j -= 1
    return out[::-1]


def align(asr_words, script, audio):
    from .asr import voiced_regions
    tokens = [t for t in script.split() if t.strip()]
    if not tokens: return []
    regions = voiced_regions(audio)
    dur = len(audio) / 16000
    if not regions: regions = [(0.0, dur)]
    anchors = _nw_anchors(tokens, [w["text"] for w in asr_words])
    times = [None] * len(tokens)
    for si, ai in anchors: times[si] = (asr_words[ai]["start"], asr_words[ai]["end"])
    # fill runs between anchors
    i = 0
    while i < len(tokens):
        if times[i] is not None: i += 1; continue
        j = i
        while j < len(tokens) and times[j] is None: j += 1
        t0 = times[i - 1][1] if i > 0 else regions[0][0]
        t1 = times[j][0] if j < len(tokens) else regions[-1][1]
        if t1 - t0 < 0.1 * (j - i):  # anchors too tight: borrow time evenly
            t1 = t0 + 0.25 * (j - i)
        for k, tt in zip(range(i, j), _spread(tokens[i:j], t0, t1, regions)): times[k] = tt
        i = j
    out = []
    for k, (tok, (s, e)) in enumerate(zip(tokens, times)):
        if out and s < out[-1]["end"]: s = out[-1]["end"]
        out.append({"text": tok, "start": round(s, 3), "end": round(max(e, s + 0.05), 3), "conf": 1.0, "approx": True, "script": True})
    return out, len(anchors) / max(1, len(tokens))
