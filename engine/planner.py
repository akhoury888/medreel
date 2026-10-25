"""Editing decision engine: analysis -> editable, reversible EditPlan (JSON).

All times in the plan are SOURCE times; the renderer maps them onto the output
timeline through the enabled cuts. Nothing ever touches the source file."""
import copy, hashlib, json, os, time, uuid
from . import knowledge as K

STYLES = {
    "clean_professional": {"label": "Clean professional", "captions": "highlight", "zoom": 0.08, "punch": 0.10, "push": 0.04,
                           "transition": 0.45, "grade": "natural", "music": "calm", "music_db": -26, "progress_bar": False, "title": "slide"},
    "modern_educational": {"label": "Modern educational", "captions": "highlight", "zoom": 0.12, "punch": 0.14, "push": 0.05,
                           "transition": 0.4, "grade": "bright", "music": "pulse", "music_db": -24, "progress_bar": True, "title": "slide"},
    "high_energy": {"label": "High-energy social", "captions": "pop", "zoom": 0.18, "punch": 0.22, "push": 0.06,
                    "transition": 0.3, "grade": "punchy", "music": "drive", "music_db": -22, "progress_bar": True, "title": "pop"},
    "cinematic_medical": {"label": "Cinematic medical", "captions": "minimal", "zoom": 0.06, "punch": 0.08, "push": 0.07,
                          "transition": 0.6, "grade": "cinematic", "music": "cinematic", "music_db": -24, "progress_bar": False, "title": "fade"},
    "minimalist_premium": {"label": "Minimalist premium", "captions": "minimal", "zoom": 0.04, "punch": 0.05, "push": 0.03,
                           "transition": 0.55, "grade": "natural", "music": "calm", "music_db": -28, "progress_bar": False, "title": "fade"},
}
CAPTION_STYLES = ["minimal", "highlight", "pop"]  # increasing dynamism
MUSIC_TRACKS = ["calm", "pulse", "drive", "cinematic"]
LAYOUTS = ["full", "side", "pip", "full_graphic", "split", "labels"]

DEFAULT_BRAND = {"name": "Dr. Antoine El Khoury", "specialty": "Interventional & Structural Cardiology", "clinic": "",
                 "logo": None, "primary": "#2EA0FF", "secondary": "#0B1730", "font": "Poppins", "website": "",
                 "booking": "", "phone": "", "intro": "none", "outro": True, "lower_third": True,
                 "disclaimer": "Educational content — not individual medical advice."}


AR_DISCLAIMER = "محتوى توعوي — لا يغني عن استشارة طبيبك."
CAPTION_MODES = ["original", "translated", "both"]  # translated = subtitles in captions.target language
FR_DISCLAIMER = "Contenu éducatif — ne remplace pas un avis médical personnalisé."
UI_TEXT = {"en": {"cta": "Book a consultation"}, "fr": {"cta": "Prendre rendez-vous"}, "ar": {"cta": "احجز موعد"}}


def _id(p): return f"{p}_{uuid.uuid4().hex[:6]}"


def build_plan(source_path, asr_backend, words, analysis, style="modern_educational", branding=None, language="en", presenter=True):
    st = STYLES[style]
    cuts = []
    for s in analysis["silences"]:
        cuts.append({"id": _id("cut"), "start": s["start"], "end": s["end"], "reason": "silence", "enabled": True, "auto": True,
                     "note": f"{s['gap']:.1f}s pause"})
    for r in analysis["retakes"]:
        cuts.append({"id": _id("cut"), "start": r["start"], "end": r["end"], "reason": "retake", "enabled": True, "auto": True,
                     "note": f"False start removed — kept the later take: “{r['kept']}”", "text": r["text"]})
    for f in analysis["fillers"]:
        sure = f.get("kind") == "hesitation"  # 'ااا' / 'euh' are never content: cut without asking
        cuts.append({"id": _id("cut"), "start": round(f["start"], 3), "end": round(f["end"], 3), "reason": "filler", "enabled": sure,
                     "auto": True, "needs_approval": not sure, "note": f"{'Hesitation' if sure else 'Filler'} “{f['text']}”"})
    cuts.sort(key=lambda c: c["start"])

    # graphics scenes — at most one per ~7 s, never overlapping, max ~50% coverage
    scenes, last_end = [], -1
    dur = analysis["meta"]["duration"]
    sents = analysis["sentences"]
    seen = {}  # (asset, mode) -> end time of its last scene
    for g in analysis["graphics_needed"]:
        gap = 1.2 if presenter else 0.0  # let the face breathe between graphics
        if g["start"] < last_end + gap:
            if g["end"] - (last_end + gap) < 2.5: continue
            g = {**g, "start": last_end + gap}  # long sentence: start its graphic part-way through
        # variety: take the best candidate not shown in the last 18 s
        cands = [g] + g.get("alts", [])
        pick = next((c for c in cands if (c["asset"], c["mode"]) not in seen or g["start"] - seen[(c["asset"], c["mode"])] >= 18.0), None)
        if pick is None:  # all shown recently: reuse the least recent one if it's been at least a few seconds
            c = min(cands, key=lambda c: seen.get((c["asset"], c["mode"]), -99))
            pick = c if g["start"] - seen.get((c["asset"], c["mode"]), -99) >= 5.0 else None
        if pick is None: continue
        g = {**g, **{k: pick[k] for k in ("asset", "mode", "priority", "layout", "label")}}
        key = (g["asset"], g["mode"])
        end = g["end"] + 0.25
        nxt = g["sentence"] + 1
        while end - g["start"] < 4.5 and nxt < len(sents):  # give animations room to play
            end = sents[nxt]["end"] + 0.25; nxt += 1
        if presenter and g["start"] < 5.0:  # opening seconds: face, hook title and name card only
            if g["end"] - 5.0 < 2.5: continue
            g = {**g, "start": 5.0}
        layout = g["layout"]
        if style in ("minimalist_premium", "clean_professional") and layout == "full_graphic": layout = "pip"
        if presenter: end = min(end, g["start"] + 8.5)  # keep the doctor on screen: no graphic hogs the frame
        scenes.append({"id": _id("scn"), "start": round(g["start"] - 0.1, 3), "end": round(min(end, dur), 3), "layout": layout,
                       "graphic": {"asset": g["asset"], "mode": g["mode"], "label": g["label"]}, "text": g["text"], "auto": True, "enabled": True})
        seen[key] = end; last_end = end
    if not presenter:  # AI-voice video: graphics fill the screen and run until the next one (max 9 s)
        for sc in scenes:  # leave the opening hook title on its own for 3.5 s
            if sc["start"] < 3.5: sc["start"] = 3.5
        scenes = [sc for sc in scenes if sc["end"] - sc["start"] > 2.0] if scenes else scenes
        for k, sc in enumerate(scenes):
            if sc["graphic"]["asset"] != "bp": sc["layout"] = "full_graphic"
            nxt = scenes[k + 1]["start"] - 0.3 if k + 1 < len(scenes) else dur
            sc["end"] = round(max(sc["end"], min(nxt, sc["start"] + 9.0)), 3)
    # close-up selfie (face fills the frame): a card beside the doctor would sit on their face
    fh = 0 if not presenter else sorted(p["box"][3] for p in analysis["face_track"])[len(analysis["face_track"]) // 2] if analysis["face_track"] else 0
    if fh > 0.26:
        for sc in scenes:
            if sc["layout"] == "side": sc["layout"] = "split"
    # vary layouts so consecutive coronary scenes don't look identical
    rot = {"side": "split", "split": "pip", "pip": "split" if fh > 0.26 else "side"}
    for a, b in zip(scenes, scenes[1:]):
        if presenter and a["layout"] == b["layout"] and b["layout"] in rot: b["layout"] = rot[b["layout"]]

    # icon cards: any specialty — concepts the doctor names (lungs, inhaler, smoking…) pop up in the gaps
    from . import concepts as CO
    pad_ = 0.8 if presenter else 0.3
    busy = [(sc["start"] - pad_, sc["end"] + pad_) for sc in scenes]
    gap_ic = 1.5 if presenter else 0.4  # with no presenter the visuals can follow each other closely
    icons, last_ic = [], -99.0
    for sen in sents:
        hits = CO.find(words[sen["i0"]: sen["i1"] + 1])
        if not hits: continue
        t0 = words[sen["i0"] + hits[0][0]]["start"] - 0.15
        if t0 < (5.0 if presenter else 3.5): continue
        if t0 - last_ic < gap_ic:
            later = [(i, k) for i, k in hits if words[sen["i0"] + i]["start"] - 0.15 >= last_ic + gap_ic]  # start at a later concept
            if not later: continue
            hits = later; t0 = words[sen["i0"] + hits[0][0]]["start"] - 0.15
        if any(a <= t0 <= b for a, b in busy): continue
        items = [{"key": k, "at": round(max(0.0, words[sen["i0"] + i]["start"] - 0.15 - t0), 2)} for i, k in hits if words[sen["i0"] + i]["start"] - t0 < 3.2][:3]
        end = max(t0 + 2.8, t0 + items[-1]["at"] + 1.8)
        end = min(end, t0 + 4.5, dur, min([a for a, b in busy if a > t0] + [dur]) - 0.2)
        if end - t0 < 2.0: continue
        icons.append({"id": _id("scn"), "start": round(t0, 3), "end": round(end, 3), "layout": "overlay",
                      "graphic": {"asset": "icons", "mode": "card", "label": " · ".join(CO.label(k, "en") for k in (it["key"] for it in items)), "items": items},
                      "text": sen["text"], "auto": True, "enabled": True})
        last_ic = end
    scenes = sorted(scenes + icons, key=lambda x: x["start"])

    # keyword emphasis punch-ins on first mention of key terms outside graphic scenes
    emph = []
    words_t = {t["start_idx"]: t for t in analysis["terms"]}
    used = set()
    for i, t in words_t.items():
        if t["label"] in used or t["label"] in K.LOW_VALUE_TERMS: continue
        w0 = words[t["start_idx"]]
        if any(s["start"] <= w0["start"] <= s["end"] for s in scenes): continue
        emph.append({"id": _id("emp"), "start": w0["start"], "end": words[t["end_idx"]]["end"] + 0.6, "term": t["label"], "enabled": True})
        used.add(t["label"])
    emph = [e for k, e in enumerate(emph) if k == 0 or e["start"] - emph[k - 1]["start"] >= 5.0][: max(4, int(analysis["meta"]["duration"] / 7))]

    return {
        "version": 1, "id": uuid.uuid4().hex[:10], "created": time.time(),
        "source": {"path": source_path, **analysis["meta"]},
        "asr": asr_backend, "style": style, "style_params": st,
        "language": language, "presenter": presenter,
        "branding": {**DEFAULT_BRAND, **({"disclaimer": AR_DISCLAIMER} if language == "ar" else {"disclaimer": FR_DISCLAIMER} if language == "fr" else {}), **(branding or {})},
        "transcript": [{"i": i, **w} for i, w in enumerate(words)],
        "analysis": {k: analysis[k] for k in ("sentences", "terms", "claims", "qa", "hook", "graphics_needed", "fillers", "retakes")},
        "face_track": analysis["face_track"],
        "cuts": cuts, "scenes": scenes, "emphasis": emph,
        "captions": {"style": st["captions"] if presenter else "pop", "enabled": True, "language": language, "max_words": None, "position": "auto",
                     "mode": "original", "target": "en", "translations": {}},
        "zoom": {"mode": "auto", "intensity": st["zoom"] if presenter else 0, "punch": st["punch"] if presenter else 0, "push": st["push"] if presenter else 0},
        "title": {"text": analysis["hook"], "enabled": bool(analysis["hook"]), "style": st["title"]},
        "music": {"track": st["music"], "enabled": True, "db": st["music_db"]},
        "audio": {"denoise": True, "loudnorm": -14, "highpass": 80},
        "grade": st["grade"],
        "output": {"width": 1080, "height": 1920, "fps": 30},
    }


def apply_style(plan, style):
    st = STYLES[style]
    plan["style"], plan["style_params"] = style, st
    plan["captions"]["style"] = st["captions"]
    plan["zoom"].update({"intensity": st["zoom"], "punch": st["punch"], "push": st["push"]})
    plan["music"].update({"track": st["music"], "db": st["music_db"]})
    plan["title"]["style"] = st["title"]; plan["grade"] = st["grade"]
    return plan


def keep_segments(plan):
    """Complement of enabled cuts -> list of (src_start, src_end)."""
    dur = plan["source"]["duration"]
    cuts = sorted([(c["start"], c["end"]) for c in plan["cuts"] if c["enabled"]])
    merged = []
    for a, b in cuts:
        a, b = max(0, a), min(dur, b)
        if b <= a: continue
        if merged and a <= merged[-1][1] + 0.05: merged[-1][1] = max(merged[-1][1], b)
        else: merged.append([a, b])
    keeps, t = [], 0.0
    for a, b in merged:
        if a - t > 0.12: keeps.append((round(t, 3), round(a, 3)))
        t = b
    if dur - t > 0.12: keeps.append((round(t, 3), round(dur, 3)))
    return keeps


class Timeline:
    """Maps source time <-> output time through kept segments."""
    def __init__(self, keeps):
        self.keeps = keeps
        self.offsets, o = [], 0.0
        for a, b in keeps: self.offsets.append(o); o += b - a
        self.duration = o

    def to_src(self, t):
        for (a, b), o in zip(self.keeps, self.offsets):
            if t < o + (b - a): return a + (t - o), self.keeps.index((a, b))
        a, b = self.keeps[-1]; return b, len(self.keeps) - 1

    def to_out(self, s, clamp=True):
        for (a, b), o in zip(self.keeps, self.offsets):
            if s < a: return o if clamp else None
            if s <= b: return o + (s - a)
        return self.duration if clamp else None

    def is_kept(self, s):
        return any(a <= s <= b for a, b in self.keeps)


# ---- persistence + undo (every change is reversible) ----
def save(plan, path, history=True):
    if history and os.path.exists(path):
        hp = path + ".history.json"
        hist = json.load(open(hp)) if os.path.exists(hp) else []
        hist.append(json.load(open(path))); hist = hist[-40:]
        json.dump(hist, open(hp, "w"))
    json.dump(plan, open(path, "w"), indent=1)


def undo(path):
    hp = path + ".history.json"
    if not os.path.exists(hp): return None
    hist = json.load(open(hp))
    if not hist: return None
    prev = hist.pop(); json.dump(hist, open(hp, "w")); json.dump(prev, open(path, "w"), indent=1)
    return prev
