"""Analysis pass: silences, fillers, retakes, medical terms, claims, engagement,
framing/lighting/audio QA and a smoothed face track for reframing."""
import json, os, re, subprocess
import numpy as np, cv2
from . import knowledge as K
from .asr import load_audio, rms_envelope, SR

MODELS = os.environ.get("MEDREEL_MODELS", os.path.join(os.path.dirname(__file__), "..", "..", "models"))


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                         capture_output=True, check=True).stdout
    j = json.loads(out)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd: rot = int(sd["rotation"])
    if abs(rot) in (90, 270): w, h = h, w
    num, den = v.get("avg_frame_rate", "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return {"width": w, "height": h, "fps": round(fps, 3), "duration": float(j["format"]["duration"]),
            "has_audio": any(s["codec_type"] == "audio" for s in j["streams"])}


def sentences(words):
    out, cur = [], []
    gap = 0.5 if any(w.get("approx") for w in words[:5]) else 1.2  # Whisper-Arabic output has little punctuation
    for i, w in enumerate(words):
        cur.append(i)
        if re.search(r"[.!?؟]$", w["text"]) or (i + 1 < len(words) and words[i + 1]["start"] - w["end"] > gap):
            out.append(cur); cur = []
    if cur: out.append(cur)
    return [{"i0": s[0], "i1": s[-1], "start": words[s[0]]["start"], "end": words[s[-1]]["end"],
             "text": " ".join(words[k]["text"] for k in s)} for s in out]


def detect_silences(words, duration, min_gap=0.45, pad=0.12):
    """Gaps between words (and head/tail) longer than min_gap. Keeps `pad` breathing room on each side."""
    gaps, prev_end = [], 0.0
    for w in words + [{"start": duration, "end": duration}]:
        g0, g1 = prev_end, w["start"]
        lead = 0.0 if prev_end == 0.0 else pad
        trail = 0.0 if w["start"] >= duration else pad
        if g1 - g0 > min_gap and (g1 - trail) - (g0 + lead) > 0.15:
            gaps.append({"start": round(g0 + lead, 3), "end": round(g1 - trail, 3), "gap": round(g1 - g0, 2)})
        prev_end = max(prev_end, w["end"])
    return gaps


def detect_fillers(words, audio):
    out = []
    toks = [K.norm(w["text"]) for w in words]
    for i, t in enumerate(toks):
        if words[i].get("approx"):
            isolated = words[i].get("region_words", 9) == 1
        else:
            isolated = (i == 0 or words[i]["start"] - words[i - 1]["end"] > 0.2) and (i + 1 == len(words) or words[i + 1]["start"] - words[i]["end"] > 0.15)
            isolated = isolated and words[i]["text"].rstrip()[-1:] in ",.،"
        if (t in K.FILLERS and (isolated or not K.is_ar(t))) or (t in K.SOFT_FILLERS and isolated):
            out.append({"start": words[i]["start"] - 0.03, "end": words[i]["end"] + 0.03, "text": words[i]["text"], "kind": "word"})
        elif i + 1 < len(toks) and (t, toks[i + 1]) in K.FILLER_PHRASES:
            out.append({"start": words[i]["start"] - 0.03, "end": words[i + 1]["end"] + 0.03, "text": words[i]["text"] + " " + words[i + 1]["text"], "kind": "phrase"})
    # untranscribed vocalisations: voiced energy inside a word gap (ASR often drops "um")
    env, hop = rms_envelope(audio)
    floor = np.percentile(env, 15); voiced = env > max(floor * 4, 0.01)
    for a, b in ([] if any(w.get("approx") for w in words[:5]) else zip(words, words[1:])):
        g0, g1 = a["end"] + 0.06, b["start"] - 0.06
        if g1 - g0 < 0.2: continue
        seg = voiced[int(g0 / hop): int(g1 / hop)]
        idx = np.where(seg)[0]
        if seg.sum() * hop > 0.10 and idx[0] >= 3 and idx[-1] <= len(seg) - 4:  # isolated from neighbouring words
            v0, v1 = g0 + idx[0] * hop, g0 + (idx[-1] + 1) * hop
            out.append({"start": round(v0 - 0.04, 3), "end": round(v1 + 0.04, 3), "text": "(um)", "kind": "vocalisation"})
    return sorted(out, key=lambda f: f["start"])


def _fuzzy_retakes(words, sents, window=15.0):
    """Sentence-level false starts: an earlier phrase that matches the opening of the next one
    (tolerant to ASR spelling drift, e.g. dialect Arabic)."""
    import difflib
    out = []
    for a, b in zip(sents, sents[1:]):
        if b["start"] - a["end"] > window or a["i1"] - a["i0"] < 2: continue
        ta = "".join(K.core(w["text"]) for w in words[a["i0"]: a["i1"] + 1])
        tb = "".join(K.core(w["text"]) for w in words[b["i0"]: b["i1"] + 1])
        if len(ta) < 8 or len(tb) <= len(ta): continue
        ra = difflib.SequenceMatcher(None, ta, tb[: len(ta) + 2]).ratio()
        last = K.core(words[a["i1"]]["text"])
        anchor = last in {K.core(w["text"]) for w in words[b["i0"]: b["i0"] + (a["i1"] - a["i0"]) + 3]}
        if ra >= 0.72 or (ra >= 0.6 and anchor and a["i1"] - a["i0"] <= 8):
            out.append({"start": round(a["start"] - 0.05, 3), "end": round(b["start"] - 0.08, 3), "text": a["text"],
                        "kept": " ".join(w["text"] for w in words[b["i0"]: b["i0"] + 6]) + "…"})
    return out


def detect_vocal_regions(words, audio):
    """Approximate-timing transcripts: short isolated speech bursts the ASR left empty are probably 'um/إمم'."""
    from .asr import voiced_regions
    out = []
    regs = voiced_regions(audio)
    for k, (r0, r1) in enumerate(regs):
        if not (0.12 <= r1 - r0 <= 0.9): continue
        pre = r0 - regs[k - 1][1] if k else 9; post = regs[k + 1][0] - r1 if k + 1 < len(regs) else 9
        inside = [w for w in words if w["start"] < r1 and w["end"] > r0]
        if inside:  # a short isolated burst that holds a real word ("شكرا", "scan") is speech, not a filler
            c = K.core(inside[0]["text"])
            if not (c in K.FILLERS or c in K.SOFT_FILLERS or len(c) <= 2): continue
        if pre >= 0.3 and post >= 0.3 and len(inside) <= 1:
            out.append({"start": round(r0 - 0.04, 3), "end": round(r1 + 0.04, 3),
                        "text": inside[0]["text"] if inside else "(إمم/um)", "kind": "vocalisation"})
    return out


def _merge_retakes(rs):
    out = []
    for r in sorted(rs, key=lambda r: r["start"]):
        if out and r["start"] <= out[-1]["end"]: out[-1]["end"] = max(out[-1]["end"], r["end"])
        else: out.append(r)
    return out


def detect_retakes(words, n=4, window=20.0):
    """A phrase of >= n words repeated within `window` s => earlier take is a false start."""
    toks = [K.core(w["text"]) for w in words]
    res, used = [], set()
    for j in range(len(toks) - n + 1):
        gram = toks[j:j + n]
        for i in range(j - 1, -1, -1):
            if words[j]["start"] - words[i]["start"] > window: break
            if toks[i:i + n] == gram and i not in used and i + n <= j:
                res.append({"start": round(words[i]["start"] - 0.05, 3), "end": round(words[j]["start"] - 0.08, 3),
                            "text": " ".join(w["text"] for w in words[i:j]),
                            "kept": " ".join(w["text"] for w in words[j:j + n + 4]) + "…"})
                used.add(i); break
    merged = []
    for r in sorted(res, key=lambda r: r["start"]):
        if merged and r["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], r["end"])
        else:
            merged.append(r)
    return merged


def engagement(sents):
    for s in sents:
        t = s["text"].lower(); sc = 0
        sc += 2 * len(K.detect_terms([{"text": x} for x in s["text"].split()]))
        ta = K.ar_norm(t)
        sc += 2 if ("?" in t or "؟" in t) else 0
        sc += 1 if re.search(r"\byou(r)?\b|انتو|انتي|عندكن|الكن|لكن|عندكم", ta) else 0
        sc += 1 if re.search(r"\d", ta) else 0
        sc += 2 if re.search(r"\b(good news|important|most|never|always|really|secret|truth)\b|الخبر المنيح|مهم|انتبهوا|شو عم يصير|ليش", ta) else 0
        sc -= 2 if re.search(r"^(hi|hello|hey)\b|^(مرحبا|اهلا|هاي|بونجور)", ta) else 0
        s["engagement"] = sc
    return sents


def hook_title(sents):
    cands = [s for s in sents[:4] if not re.match(r"^(hi|hello|hey)\b|^(مرحبا|اهلا|هاي)", K.ar_norm(s["text"].lower()))] or sents[:1]
    s = max(cands, key=lambda s: s.get("engagement", 0)) if cands else {"text": ""}
    t = re.sub(r"^(today,? )?(i want to|i'm going to|let me|we will|we'll) (explain|talk about|show you|tell you)( about)? ", "", s["text"], flags=re.I)
    t = re.sub(r"^(اليوم,?\s+)?(بدي|حابب|رح|حابه)\s+(إ|ا)?(شرحلكن|شرحلكم|حكيلكن|حكيلكم|خبركن|فرجيكن)\s+(عن\s+)?", "", t)
    t = re.sub(r"[.!،,]$", "", t).strip()
    t = t[0].upper() + t[1:] if t else t
    words = t.split()
    return " ".join(words[:9]) + ("…" if len(words) > 9 else "")


def face_track(path, meta, step=0.5):
    det = None
    yp = os.path.join(MODELS, "yunet.onnx")
    if os.path.exists(yp):
        det = cv2.FaceDetectorYN.create(yp, "", (320, 320), 0.7)
    sw = 480; sh = int(round(meta["height"] * sw / meta["width"] / 2) * 2)
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={1/step},scale={sw}:{sh}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (sw * sh * 3)
    frames = np.frombuffer(raw[: n * sw * sh * 3], np.uint8).reshape(n, sh, sw, 3)
    pts, lum, sharp, con = [], [], [], []
    for k, f in enumerate(frames):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); lum.append(float(g.mean())); con.append(float(g.std())); sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        box = None
        if det is not None:
            det.setInputSize((sw, sh)); _, faces = det.detect(f)
            if faces is not None and len(faces):
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])[:4]
                box = [float(x / sw), float(y / sh), float(w / sw), float(h / sh)]
        pts.append({"t": round(k * step, 2), "box": box})
    detected = [p for p in pts if p["box"]]
    if detected:  # fill gaps + EMA smoothing
        last = detected[0]["box"]
        for p in pts:
            if p["box"]: last = p["box"]
            else: p["box"] = last; p["interp"] = True
        sm = None
        for p in pts:
            sm = p["box"] if sm is None else [0.75 * a + 0.25 * b for a, b in zip(sm, p["box"])]
            p["box"] = [round(v, 4) for v in sm]
    else:  # heuristic: presenter head in upper-centre third
        for p in pts: p["box"] = [0.36, 0.16, 0.28, 0.17]; p["interp"] = True
    return pts, len(detected) / max(1, len(pts)), float(np.mean(lum)) if lum else 0, float(np.mean(con)) if con else 0, float(np.median(sharp)) if sharp else 0


def audio_qa(audio):
    env, hop = rms_envelope(audio)
    db = 20 * np.log10(env + 1e-9)
    speech = db[db > np.percentile(db, 60)]
    noise = db[db < np.percentile(db, 10)]
    clip = float((np.abs(audio) > 0.98).mean())
    return {"speech_dbfs": round(float(np.median(speech)), 1), "noise_floor_dbfs": round(float(np.median(noise)), 1),
            "snr_db": round(float(np.median(speech) - np.median(noise)), 1), "clipping_pct": round(clip * 100, 3)}


def qa_report(meta, face_ratio, face_pts, lum_mean, lum_std, sharp, aqa):
    issues = []
    if meta["width"] > meta["height"]:
        issues.append({"area": "framing", "level": "info", "msg": "Landscape recording — auto-reframed to 9:16 around your face."})
    if face_ratio < 0.5:
        issues.append({"area": "framing", "level": "warn", "msg": f"Face detected in only {face_ratio*100:.0f}% of frames — framing uses a fallback position."})
    else:
        cx = np.mean([p["box"][0] + p["box"][2] / 2 for p in face_pts]); top = np.mean([p["box"][1] for p in face_pts])
        if abs(cx - 0.5) > 0.15: issues.append({"area": "framing", "level": "info", "msg": "Face is off-centre — reframing will re-centre it."})
        if top < 0.05: issues.append({"area": "framing", "level": "warn", "msg": "Very little headroom — keep your eyes about 1/3 from the top."})
    if lum_mean < 70: issues.append({"area": "lighting", "level": "warn", "msg": "Footage is dark — brightness lifted automatically; face a window or add a key light next time."})
    elif lum_mean > 190: issues.append({"area": "lighting", "level": "warn", "msg": "Footage is very bright/overexposed."})
    if lum_std < 30: issues.append({"area": "lighting", "level": "info", "msg": "Low contrast — contrast enhanced."})
    if sharp < 30: issues.append({"area": "image", "level": "info", "msg": "Image looks soft — check focus / clean the lens."})
    if aqa["snr_db"] < 20: issues.append({"area": "audio", "level": "warn", "msg": f"Background noise is high (SNR {aqa['snr_db']} dB) — noise reduction applied; a lav mic helps."})
    if aqa["clipping_pct"] > 0.05: issues.append({"area": "audio", "level": "warn", "msg": "Audio clipping detected — record a little further from the mic."})
    if aqa["speech_dbfs"] < -35: issues.append({"area": "audio", "level": "info", "msg": "Voice level is low — normalised to -14 LUFS."})
    score = max(0, 100 - 12 * sum(i["level"] == "warn" for i in issues) - 4 * sum(i["level"] == "info" for i in issues))
    return {"score": score, "issues": issues, "luma_mean": round(lum_mean, 1), "luma_std": round(lum_std, 1),
            "sharpness": round(sharp, 1), "audio": aqa, "face_detect_ratio": round(face_ratio, 2)}


def _minus(sils, fillers, pad=0.08):
    """Silence cuts must never swallow a vocalisation (that would remove a filler without approval)."""
    out = []
    for s in sils:
        parts = [(s["start"], s["end"])]
        for f in fillers:
            nxt = []
            for a, b in parts:
                if f["end"] <= a or f["start"] >= b: nxt.append((a, b)); continue
                if f["start"] - pad - a > 0.2: nxt.append((a, f["start"] - pad))
                if b - (f["end"] + pad) > 0.2: nxt.append((f["end"] + pad, b))
            parts = nxt
        out += [{"start": round(a, 3), "end": round(b, 3), "gap": s["gap"]} for a, b in parts]
    return out


def analyze(path, words, lang="en", extra_retakes=None, presenter=True):
    meta = probe(path)
    audio = load_audio(path)
    sents = engagement(sentences(words))
    terms = K.detect_terms(words)
    claims = []
    for s in sents:
        rs = K.flag_claims(s["text"])
        if rs: claims.append({"start": s["start"], "end": s["end"], "text": s["text"], "reason": "; ".join(rs)})
    graphics_needed = []
    for idx, s in enumerate(sents):
        m = K.match_graphic(s["text"])
        if m: graphics_needed.append({"sentence": idx, "start": s["start"], "end": s["end"], **m, "text": s["text"]})
    fillers = detect_fillers(words, audio) if presenter else []
    if presenter and any(w.get("approx") for w in words[:5]):
        seen = {(round(f["start"], 1)) for f in fillers}
        fillers += [f for f in detect_vocal_regions(words, audio) if round(f["start"], 1) not in seen]
        fillers.sort(key=lambda f: f["start"])
    pts, fr, lm, ls, sh = face_track(path, meta) if presenter else ([], 0, 110, 60, 100)
    aqa = audio_qa(audio)
    return {
        "language": lang,
        "meta": meta, "sentences": sents, "silences": _minus(detect_silences(words, meta["duration"]), fillers),
        "fillers": fillers, "retakes": _merge_retakes(detect_retakes(words) + _fuzzy_retakes(words, sents) + list(extra_retakes or [])), "terms": terms,
        "claims": claims, "graphics_needed": graphics_needed, "face_track": pts,
        "qa": qa_report(meta, fr, pts, lm, ls, sh, aqa) if presenter else {"score": 100, "issues": [], "luma_mean": 110, "luma_std": 60,
              "sharpness": 100, "audio": aqa, "face_detect_ratio": 0}, "hook": hook_title(sents),
    }
