"""Speech-to-text with word-level timestamps.

Backends (auto-selected):
  * parakeet  — NVIDIA Parakeet-TDT 0.6B v2 via sherpa-onnx (CC-BY-4.0 weights, Apache-2.0 runtime). CPU, ~7x realtime.
  * faster-whisper — Whisper via CTranslate2 (MIT). Use for multilingual (FR/AR/ES...).
Output: list of {text, start, end, conf}.
"""
import os, subprocess, numpy as np

MODELS = os.environ.get("MEDREEL_MODELS", os.path.join(os.path.dirname(__file__), "..", "..", "models"))
PARAKEET_DIR = os.path.join(MODELS, "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8")
SR = 16000


def load_audio(path, sr=SR):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32)


def rms_envelope(a, sr=SR, hop=0.02):
    n = int(sr * hop)
    m = len(a) // n
    return np.sqrt((a[: m * n].reshape(m, n) ** 2).mean(1) + 1e-12), hop


def _chunks(a, sr=SR, target=22.0, maxlen=28.0):
    env, hop = rms_envelope(a, sr)
    total = len(a) / sr
    cuts, t = [0.0], 0.0
    while total - t > maxlen:
        lo, hi = int((t + target - 6) / hop), int((t + maxlen) / hop)
        idx = lo + int(np.argmin(env[lo:hi]))
        t = idx * hop
        cuts.append(t)
    cuts.append(total)
    return list(zip(cuts[:-1], cuts[1:]))


_parakeet = None


def _parakeet_rec():
    global _parakeet
    if _parakeet is None:
        import sherpa_onnx
        d = PARAKEET_DIR + "/"
        _parakeet = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=d + "encoder.int8.onnx", decoder=d + "decoder.int8.onnx", joiner=d + "joiner.int8.onnx",
            tokens=d + "tokens.txt", model_type="nemo_transducer", num_threads=max(1, os.cpu_count() or 1))
    return _parakeet


def _transcribe_parakeet(a):
    rec = _parakeet_rec()
    words = []
    for s0, s1 in _chunks(a):
        seg = a[int(s0 * SR): int(s1 * SR)]
        st = rec.create_stream(); st.accept_waveform(SR, seg); rec.decode_stream(st)
        r = st.result
        durs = list(getattr(r, "durations", []) or [])
        cur = None
        for k, (tok, ts) in enumerate(zip(r.tokens, r.timestamps)):
            d = durs[k] if k < len(durs) and durs[k] > 0 else 0.08
            d = min(d, 0.32)  # TDT can attach trailing silence to a token's duration
            t0, t1 = s0 + ts, s0 + ts + d
            if not any(ch.isalnum() for ch in tok):  # punctuation never extends a word
                t1 = cur["end"] if cur else t1
            new_word = tok.startswith(" ") or tok.startswith("▁") or cur is None
            piece = tok.replace("▁", " ")
            if new_word and piece.strip() and piece.strip()[0].isalnum():
                if cur: words.append(cur)
                cur = {"text": piece.strip(), "start": t0, "end": t1, "conf": 1.0}
            else:
                if cur is None:
                    cur = {"text": piece.strip(), "start": t0, "end": t1, "conf": 1.0}
                else:
                    cur["text"] += piece.strip() if not piece.startswith(" ") else piece
                    cur["end"] = max(cur["end"], t1)
        if cur: words.append(cur)
    for w in words:
        w["text"] = w["text"].strip(); w["start"] = round(w["start"], 3); w["end"] = round(w["end"], 3)
    return [w for w in words if w["text"]]


def _transcribe_whisper(path, language=None, model="small"):
    from faster_whisper import WhisperModel
    m = WhisperModel(model, device="auto", compute_type="int8")
    segs, _ = m.transcribe(path, word_timestamps=True, language=language, vad_filter=False,
                           initial_prompt="Um, uh, so, cardiology: coronary arteries, angioplasty, stent, atherosclerosis.")
    return [{"text": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3), "conf": round(w.probability, 3)}
            for s in segs for w in s.words]


# ---------------- multilingual: Whisper (sherpa-onnx export) + audio-based word alignment ----------------
WHISPER_DIRS = {"turbo": "sherpa-onnx-whisper-turbo", "large-v3": "sherpa-onnx-whisper-large-v3"}
_whisper = {}


def _whisper_rec(lang, name=None):
    import glob, sherpa_onnx
    name = name or os.environ.get("MEDREEL_WHISPER", "turbo")
    if not os.path.isdir(os.path.join(MODELS, WHISPER_DIRS[name])): name = next(n for n, d in WHISPER_DIRS.items() if os.path.isdir(os.path.join(MODELS, d)))
    if (name, lang) not in _whisper:
        d = os.path.join(MODELS, WHISPER_DIRS[name])
        g = lambda pat: glob.glob(os.path.join(d, pat))[0]
        _whisper[(name, lang)] = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=g("*encoder.int8.onnx"), decoder=g("*decoder.int8.onnx"), tokens=g("*tokens.txt"),
            language=lang or "", task="transcribe", num_threads=max(1, os.cpu_count() or 1))
    return _whisper[(name, lang)], name


def whisper_available():
    return any(os.path.isdir(os.path.join(MODELS, d)) for d in WHISPER_DIRS.values())


def voiced_regions(a, sr=SR, min_gap=0.2, min_len=0.08):
    """Energy VAD -> [(t0, t1)] speech regions separated by >= min_gap s of quiet."""
    env, hop = rms_envelope(a, sr)
    thr = max(np.percentile(env, 10) * 3.5, np.percentile(env, 60) * 0.12, 0.006)
    v = env > thr
    regs, i, n = [], 0, len(v)
    while i < n:
        if v[i]:
            j = i
            while j < n and v[j]: j += 1
            regs.append([i * hop, j * hop]); i = j
        else: i += 1
    merged = []
    for r in regs:
        if merged and r[0] - merged[-1][1] < min_gap: merged[-1][1] = r[1]
        else: merged.append(r)
    return [(round(a0, 3), round(b0, 3)) for a0, b0 in merged if b0 - a0 >= min_len]


def _decode(rec, seg):
    st = rec.create_stream(); st.accept_waveform(SR, seg); rec.decode_stream(st)
    return st.result


def detect_language(a):
    rec, _ = _whisper_rec("")
    regs = voiced_regions(a)
    if not regs: return "en"
    s0 = regs[0][0]; s1 = min(len(a) / SR, s0 + 20)
    r = _decode(rec, a[int(s0 * SR): int(s1 * SR)])
    return (getattr(r, "lang", "") or "en").strip("<|>") or "en"


def _transcribe_whisper_sherpa(a, lang, name=None, context=None):
    """Whisper gives text per chunk but (in this export) no word times, so words are aligned to the
    energy-VAD speech regions by character mass: pauses between regions are exact, word times inside
    a region are approximate (+-0.2 s) — good for captions and cut decisions."""
    from . import knowledge as K
    rec, name = _whisper_rec(lang, name)
    regs = voiced_regions(a); dur = len(a) / SR
    # decode phrase by phrase (split at pauses >= 0.4 s) so words can't drift across a pause
    chunks, cur = [], []
    for r in regs:
        if cur and (r[0] - cur[-1][1] >= 0.4 or r[1] - cur[0][0] > 24): chunks.append(cur); cur = []
        cur.append(r)
    if cur: chunks.append(cur)
    texts = []
    for ch in chunks:
        s0, s1 = max(0, ch[0][0] - 0.15), min(dur, ch[-1][1] + 0.15)
        texts.append(_decode(rec, a[int(s0 * SR): int(s1 * SR)]).text.strip())
    if os.environ.get("ANTHROPIC_API_KEY") and lang in ("ar",) and context is not False:
        alts = None
        other = next((n for n in WHISPER_DIRS if n != name and os.path.isdir(os.path.join(MODELS, WHISPER_DIRS[n]))), None)
        if other and name == "large-v3":  # a second opinion: the two models mishear different words
            rec2, _ = _whisper_rec(lang, other)
            alts = [_decode(rec2, a[int(max(0, ch[0][0] - 0.15) * SR): int(min(dur, ch[-1][1] + 0.15) * SR)]).text.strip() for ch in chunks]
        try:
            from .translate import clean_transcript
            texts = clean_transcript(texts, lang, alts=alts, context=context)
            name = name + "+ai"
        except Exception:
            pass  # cleanup is best-effort; raw ASR text is still usable and editable
    words, rid = [], 0
    for ch, text in zip(chunks, texts):
        toks = [t for t in text.split() if t.strip()]
        if not toks: rid += len(ch); continue
        wts = [len(K.core(t)) + 1 for t in toks]; W = float(sum(wts))
        D = [b - a_ for a_, b in ch]; Dt = sum(D)
        bounds = np.cumsum([0] + D) / Dt
        assign, acc = [], 0.0
        for wt in wts:
            mid = (acc + wt / 2) / W; acc += wt
            assign.append(int(min(len(ch) - 1, np.searchsorted(bounds, mid, side="right") - 1)))
        for k, (r0, r1) in enumerate(ch):
            idx = [i for i, x in enumerate(assign) if x == k]
            if not idx: continue
            tot = sum(wts[i] for i in idx); t = r0
            for i in idx:
                d = (r1 - r0) * wts[i] / tot
                words.append({"text": toks[i], "start": round(t, 3), "end": round(t + d, 3), "conf": 1.0,
                              "region": rid + k, "region_words": len(idx), "approx": True})
                t += d
        rid += len(ch)
    if lang == "ar" or any(K.is_ar(w["text"]) for w in words[:20]):
        words = K.correct_medical_terms(words)
    return f"whisper-{name}", words


def transcribe(path, language="auto", quality="best", context=None):
    """-> (backend, words, language). quality='best' uses Whisper large-v3 for Arabic (much better on
    Lebanese dialect, ~6x slower on CPU — use a GPU worker in production); 'fast' uses turbo (timing only)."""
    backend = os.environ.get("MEDREEL_ASR", "auto")
    a = load_audio(path)
    if language in (None, "", "auto"):
        language = detect_language(a) if whisper_available() else "en"
    if backend in ("auto", "parakeet") and language == "en" and os.path.isdir(PARAKEET_DIR):
        return "parakeet-tdt-0.6b-v2", _transcribe_parakeet(a), "en"
    if backend in ("auto", "whisper") and whisper_available():
        name = "large-v3" if (quality == "best" and language == "ar" and os.environ.get("MEDREEL_WHISPER_AR", "large-v3") == "large-v3") else None
        b, w = _transcribe_whisper_sherpa(a, language, name, context if quality == "best" else False)
        return b, w, language
    return "faster-whisper", _transcribe_whisper(path, language), language
