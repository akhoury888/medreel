"""Natural-language editing -> structured, reversible plan operations.

Two interpreters:
  1. Deterministic rule parser (offline, instant, covers the common agency requests).
  2. Optional LLM interpreter (Claude via ANTHROPIC_API_KEY) used when rules don't match;
     it may only emit ops from OPS_SCHEMA, which are validated before applying.
Every command is saved with an undo snapshot (planner.save)."""
import json, os, re, uuid
from . import planner as P, knowledge as K

OPS_SCHEMA = {
    "set_style": {"style": list(P.STYLES)}, "caption_dynamism": {"delta": "int"}, "set_caption_style": {"style": P.CAPTION_STYLES},
    "captions_enabled": {"enabled": "bool"}, "cut_range": {"start_out": "sec", "end_out": "sec"}, "remove_pause_near": {"t_out": "sec"},
    "cut_phrase": {"phrase": "str"}, "enable_cuts": {"reason": "silence|retake|filler"}, "approve_fillers": {}, "restore_all_cuts": {"reason": "silence|retake|filler|manual|any"},
    "add_graphic": {"asset": "coronary/narrowing|coronary/occlusion|coronary/stent|ecg/sinus|bp/gauge|ct/calcium|ct/scanner|heart/coronaries", "phrase": "str?", "t_out": "sec?", "layout": "layout?"},
    "remove_graphics": {"asset": "str?"}, "set_layout": {"asset": "str?", "layout": P.LAYOUTS}, "face_visible": {},
    "set_music": {"track": P.MUSIC_TRACKS + ["next", "off", "on"]}, "music_volume": {"delta_db": "int"},
    "zoom": {"delta": "float"}, "caption_mode": {"mode": "original|translated|both", "target": "en|fr|ar"},
    "translate_captions": {"target": "en|fr|ar"}, "set_translation": {"i": "sentence index", "text": "str"}, "set_title": {"text": "str"}, "title_enabled": {"enabled": "bool"}, "undo": {},
    "add_broll": {"clip": "id", "sentence": "int?"}, "image_style": {"style": "3d|flat|photo"}, "add_photo": {"photo": "id", "sentence": "int?", "phrase": "str?", "t_out": "sec?", "caption": "str?"}, "delete_scene": {"id": "str"}, "scene_time": {"id": "str", "start_out": "sec", "end_out": "sec"}, "set_effects": {"transitions": "bool?", "sfx": "bool?"},
}

ASSET_WORDS = [("coronary/stent", r"stent|angioplast|balloon"), ("coronary/occlusion", r"clot|heart attack|infarct|ruptur|occlu"),
               ("coronary/narrowing", r"plaque|cholesterol|narrow|atheroscl|coronary|artery|arteries|heart animation"),
               ("ct/calcium", r"calcium|agatston|كالسيوم|كلس"), ("ct/scanner", r"ct scan|scanner|سكانر|coroscanner"),
               ("heart/coronaries", r"heart|coronary tree|coronaries|قلب"), ("ecg/sinus", r"\becg\b|\bekg\b|rhythm|electrocardio"), ("bp/gauge", r"blood pressure|\bbp\b")]
LAYOUT_WORDS = [("pip", r"picture.in.picture|\bpip\b|bubble|circle|corner"), ("split", r"split"), ("side", r"beside|side by side|next to me|card"),
                ("full_graphic", r"full.?screen (animation|graphic)|only the animation|hide (me|my face)"), ("labels", r"label|overlay"), ("full", r"just me|no graphic")]


def _sec(txt):
    m = re.search(r"(\d+):(\d{1,2})", txt)
    if m: return int(m.group(1)) * 60 + int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:s\b|sec|second)", txt)
    if m: return float(m.group(1))
    m = re.search(r"\bat (\d+(?:\.\d+)?)\b", txt)
    return float(m.group(1)) if m else None


def _asset(txt):
    for a, pat in ASSET_WORDS:
        if re.search(pat, txt): return a
    return None


def _layout(txt):
    for l, pat in LAYOUT_WORDS:
        if re.search(pat, txt): return l
    return None


def parse_rules(cmd):
    c = cmd.lower().strip()
    ops = []
    if re.fullmatch(r"(undo|revert|go back)( that| the last (change|edit))?\.?", c): return [{"op": "undo"}]
    lang_word = re.search(r"\b(english|french|arabic|anglais|fran[cç]ais)\b|انكليزي|إنكليزي|انجليزي|فرنسي|عربي", c)
    if lang_word and re.search(r"subtitle|caption|translat|ترجم|الترجمه|الترجمة|sous-titre", c):
        found = re.findall(r"\b(?:english|french|arabic|anglais|fran[cç]ais)\b|انكليزي|إنكليزي|انجليزي|فرنسي|عربي", c)
        w = next((x for x in found if not re.search(r"arabic|عربي", x)), found[0])
        tgt = "en" if re.search(r"english|anglais|كليزي|جليزي", w) else "fr" if re.search(r"french|fran|فرنسي", w) else "ar"
        both = bool(re.search(r"\bboth\b|\band\b.*\b(arabic|english|french)\b|bilingual|الاثنين|التنين|مع ", c))
        return [{"op": "caption_mode", "mode": "original"}] if tgt == "ar" and not both else [{"op": "translate_captions", "target": tgt}, {"op": "caption_mode", "mode": "both" if both else "translated", "target": tgt}]
    if "subtitle" in c or "caption" in c:
        if re.search(r"more (dynamic|energetic|lively|fun|animated)|bolder|punchier", c): ops.append({"op": "caption_dynamism", "delta": 1})
        elif re.search(r"less (dynamic|busy)|calmer|simpler|minimal|subtle|cleaner", c): ops.append({"op": "caption_dynamism", "delta": -1})
        elif re.search(r"remove|turn off|hide|no ", c): ops.append({"op": "captions_enabled", "enabled": False})
        elif re.search(r"turn on|add|show", c): ops.append({"op": "captions_enabled", "enabled": True})
        for st in P.CAPTION_STYLES:
            if re.search(rf"\b{st}\b", c) and not ops: ops.append({"op": "set_caption_style", "style": st})
    if re.search(r"(remove|cut|delete|shorten|tighten).*(pause|silence|gap|dead air)", c):
        t = _sec(c)
        ops.append({"op": "remove_pause_near", "t_out": t} if t is not None else {"op": "enable_cuts", "reason": "silence"})
    if re.search(r"(remove|cut|delete).*\b(um+s?|uh+s?|filler)", c) or re.search(r"approve.*filler", c): ops.append({"op": "approve_fillers"})
    m = re.search(r"(?:remove|cut|delete|drop) (?:the )?(?:sentence|part|bit|line|claim|section)?\s*(?:about|where i say|that says|saying|with)\s+[\"“']?(.+?)[\"”']?\.?$", c)
    if m and not ops: ops.append({"op": "cut_phrase", "phrase": m.group(1)})
    m = re.search(r"(?:cut|remove|delete) (?:from )?(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)? (?:to|-|until) (\d+(?:\.\d+)?)", c)
    if m: ops.append({"op": "cut_range", "start_out": float(m.group(1)), "end_out": float(m.group(2))})
    if re.search(r"\b(add|insert|show|put)\b.*\b(animation|graphic|illustration|visual|diagram)\b", c):
        a = _asset(c) or "coronary/narrowing"
        m = re.search(r"when i (?:discuss|talk about|mention|say|explain|speak about)\s+(.+?)\.?$", c) or re.search(r"(?:during|at) (?:the part about )?(.+?)\.?$", c)
        op = {"op": "add_graphic", "asset": a, "layout": _layout(c)}
        t = _sec(c)
        if t is not None: op["t_out"] = t
        elif m: op["phrase"] = m.group(1)
        ops.append(op)
    elif re.search(r"(remove|delete|no) .*(animation|graphic)s?", c):
        ops.append({"op": "remove_graphics", "asset": _asset(c)})
    if re.search(r"(keep|show|make).*(face|me).*(visible|on screen|seen)|don.t hide (me|my face)", c): ops.append({"op": "face_visible"})
    elif re.search(r"\b(use|make|switch|change)\b.*\b(layout|split|picture.in.picture|pip|side by side|full.?screen)\b", c):
        l = _layout(c)
        if l: ops.append({"op": "set_layout", "asset": _asset(c), "layout": l})
    if "music" in c:
        if re.search(r"(no|remove|turn off|mute|without) .*music|music off", c): ops.append({"op": "set_music", "track": "off"})
        elif re.search(r"(lower|quieter|softer|reduce)", c): ops.append({"op": "music_volume", "delta_db": -4})
        elif re.search(r"(louder|higher|increase|raise)", c): ops.append({"op": "music_volume", "delta_db": 3})
        else:
            tr = next((t for t in P.MUSIC_TRACKS if t in c), None)
            tr = tr or ("drive" if re.search(r"energ|upbeat|fast", c) else "cinematic" if "cinematic" in c or "epic" in c else "calm" if re.search(r"calm|soft|relax", c) else "next")
            ops.append({"op": "set_music", "track": tr})
    if re.search(r"more professional|more serious|corporate|cleaner look|polished", c) and not ops: ops.append({"op": "set_style", "style": "clean_professional"})
    for key, st in (("high.?energy|viral|tiktok|more energetic", "high_energy"), ("cinematic", "cinematic_medical"), ("minimal(ist)?|premium|luxur", "minimalist_premium"),
                    ("educational|modern", "modern_educational")):
        if re.search(rf"(style|look|feel|make it|make the video).*({key})|({key}).*(style|look)", c) and not any(o["op"] == "set_style" for o in ops) and "caption" not in c and "subtitle" not in c and "music" not in c:
            ops.append({"op": "set_style", "style": st}); break
    if re.search(r"(more|bigger|stronger) zoom|zoom (in )?more", c): ops.append({"op": "zoom", "delta": 0.06})
    elif re.search(r"(less|smaller|subtle|no) zoom|zoom less|stop zooming", c): ops.append({"op": "zoom", "delta": -0.06})
    m = re.search(r"(?:change|set|make) (?:the )?title (?:to|:)\s*[\"“']?(.+?)[\"”']?$", cmd.strip(), re.I)
    if m: ops.append({"op": "set_title", "text": m.group(1)})
    elif re.search(r"(remove|hide|no) (the )?title", c): ops.append({"op": "title_enabled", "enabled": False})
    return ops


OP_HELP = """set_style: overall editing look (colours, zoom, music, subtitle style). caption_dynamism: +1 livelier / -1 calmer subtitles.
set_caption_style: minimal|highlight|pop. captions_enabled: subtitles on/off. cut_range: remove a part (preview seconds).
remove_pause_near: remove the pause near a time. cut_phrase: remove the sentence containing a phrase. enable_cuts: auto-remove all silences/retakes/fillers.
restore_all_cuts: put removed parts back. add_graphic: built-in medical animation at a phrase or time. remove_graphics: remove animations (asset optional).
set_layout: how an animation is shown with the doctor (asset optional = all). face_visible: never hide the doctor's face.
set_music: track calm|pulse|drive|cinematic, or next/off/on. music_volume: delta_db (+3 louder, -4 quieter).
zoom: delta (+0.06 more camera zoom, -0.06 less). translate_captions + caption_mode: subtitles in another language (both = original + translation).
set_title / title_enabled: the opening title. image_style: AI illustration style. set_effects: transitions = motion graphics on/off, sfx = sound effects on/off.
delete_scene / scene_time: remove or retime a visual by id (times in preview seconds)."""


def parse_llm(cmd, plan):
    """Returns (ops, reply). reply explains in one friendly sentence when the request can't be done."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key: return [], "Free-text requests need the Claude key (ANTHROPIC_API_KEY) — the buttons above still work."
    import urllib.request, logging
    tl = P.Timeline(P.keep_segments(plan))
    sents = [f"[{tl.to_out(s['start']):.1f}s] {s['text']}" for s in plan["analysis"]["sentences"]]
    scenes = [{"id": s["id"], "from": round(tl.to_out(s["start"]), 1), "to": round(tl.to_out(s["end"]), 1), "layout": s.get("layout"),
               "asset": s.get("graphic", {}).get("asset"), "about": (s.get("text") or s.get("caption") or "")[:60],
               "hidden": not s.get("enabled", True)} for s in plan["scenes"]]
    state = {"style": plan.get("style"), "music": plan.get("music"), "captions": {k: plan["captions"].get(k) for k in ("style", "enabled", "mode")},
             "title": plan["title"].get("text"), "effects": plan.get("effects")}
    prompt = (f"You turn a doctor's request about their social-media video into editing operations.\n"
              f"Allowed ops and parameters (JSON): {json.dumps(OPS_SCHEMA)}\nWhat they do:\n{OP_HELP}\n"
              f"All times are seconds in the edited video.\nCurrent settings: {json.dumps(state, ensure_ascii=False)}\n"
              f"Visuals on the timeline: {json.dumps(scenes, ensure_ascii=False)}\nScript (edited-video times):\n" + "\n".join(sents) +
              f"\n\nRequest (may be in English, French or Lebanese Arabic): {cmd}\n\n"
              "Reply with ONLY a JSON object: {\"ops\": [{\"op\": \"...\", ...}], \"reply\": \"...\"}. "
              "Pick the closest ops even if the request is loosely worded. If nothing fits, return no ops and put in reply one short, "
              "friendly sentence (in the request's language) saying what you can't do and what the doctor can do instead in the studio.")
    body = json.dumps({"model": os.environ.get("MEDREEL_LLM", "claude-sonnet-5-5"), "max_tokens": 1000,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body, {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        res = json.loads(urllib.request.urlopen(req, timeout=60).read())
        txt = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        obj = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
        ops = [o for o in obj.get("ops", []) if isinstance(o, dict) and o.get("op") in OPS_SCHEMA]
        return ops, (obj.get("reply") or "").strip()
    except Exception as e:
        detail = ""
        if hasattr(e, "read"):
            try: detail = e.read().decode()[:300]
            except Exception: pass
        logging.getLogger("medreel.commands").warning("Claude edit request failed: %s %s", e, detail)
        if "401" in str(e) or "authentication" in detail: return [], "The Claude key was refused — check ANTHROPIC_API_KEY in your .env file."
        if "credit" in detail.lower() or "billing" in detail.lower(): return [], "The Claude account has no credit left — add credit at console.anthropic.com."
        return [], "I couldn't reach Claude just now — please try again in a moment."


def _find_phrase(plan, phrase):
    words = plan["transcript"]; toks = [K.norm(w["text"]) for w in words]
    q = [K.norm(x) for x in re.sub(r"^(the|a|an|my|about)\s+", "", phrase.lower()).split() if K.norm(x)]
    if not q: return None
    for i in range(len(toks)):
        if all(j < len(toks) and (toks[i + k].startswith(q[k][:max(4, len(q[k]) - 2)]) if len(q[k]) > 3 else toks[i + k] == q[k])
               for k, j in enumerate(range(i, i + len(q)))):
            return i
    for i, t in enumerate(toks):  # fallback: any word stem
        if any(len(x) > 4 and t.startswith(x[:5]) for x in q): return i
    return None


def _sentence_of(plan, idx):
    for s in plan["analysis"]["sentences"]:
        if s["i0"] <= idx <= s["i1"]: return s
    return None


def _edit_sentence(plan, si, text):
    """Retype a whole sentence (the fast way to fix dialect ASR). Timing is redistributed across the
    sentence's original span by character weight, so cuts, scenes and captions stay in sync."""
    sents = plan["analysis"]["sentences"]; s = sents[si]
    toks = [t for t in text.split() if t.strip()]
    if not toks: return "Empty sentence — use a cut to remove it instead."
    old = plan["transcript"][s["i0"]: s["i1"] + 1]
    t0, t1 = old[0]["start"], old[-1]["end"]
    wts = [len(K.core(t)) + 1 for t in toks]; tot = float(sum(wts)); t = t0; new = []
    for tok, wt in zip(toks, wts):
        d = (t1 - t0) * wt / tot
        new.append({"text": tok, "start": round(t, 3), "end": round(t + d, 3), "conf": 1.0, "edited": True, "approx": True}); t += d
    tr = plan["transcript"][: s["i0"]] + new + plan["transcript"][s["i1"] + 1:]
    for i, w in enumerate(tr): w["i"] = i
    plan["transcript"] = tr
    delta = len(new) - len(old)
    for k, x in enumerate(sents):
        if k == si: x["i1"] = x["i0"] + len(new) - 1; x["text"] = " ".join(toks)
        elif k > si: x["i0"] += delta; x["i1"] += delta
    plan["analysis"]["terms"] = K.detect_terms(tr)
    plan["analysis"]["claims"] = [c for c in plan["analysis"]["claims"] if not (abs(c["start"] - s["start"]) < 0.01)]
    rs = K.flag_claims(s["text"])
    if rs: plan["analysis"]["claims"].append({"start": s["start"], "end": s["end"], "text": s["text"], "reason": "; ".join(rs)})
    plan.get("captions", {}).get("translations", {}).pop(str(si), None)  # stale translation of the old wording
    return f"Line {si + 1} corrected."


def _paid(sc):
    g = sc.get("graphic", {})
    return bool(g.get("fill") or g.get("animated") or (g.get("asset") == "photo"))


def _make_room(scenes, a, b, keep_icons=False):
    """Clear [a, b] on the visuals track: neighbours are trimmed when enough of them is left (>= 0.8 s); one that is
    fully covered is removed — unless it's an AI image/animation or the doctor's photo, which is hidden (kept on the
    timeline, dimmed) so paid or personal content is never lost by a drag."""
    out = []
    for x in scenes:
        ov = min(x["end"], b) - max(x["start"], a)
        if ov <= 0: out.append(x); continue
        if x.get("layout") == "overlay" or x.get("graphic", {}).get("asset") == "icons":
            if keep_icons: out.append(x)
            continue  # icon cards give way to a real visual
        if x["start"] < a and a - x["start"] >= 0.8: x["end"] = round(a - 0.02, 3); out.append(x)
        elif x["end"] > b and x["end"] - b >= 0.8: x["start"] = round(b + 0.02, 3); out.append(x)
        elif _paid(x): x["enabled"] = False; out.append(x)
    return out


def apply(plan, op):
    o = op["op"]; tl = P.Timeline(P.keep_segments(plan))
    if o == "set_style":
        P.apply_style(plan, op["style"]); return f"Applied the “{P.STYLES[op['style']]['label']}” style (cuts, graphics and branding kept)."
    if o == "caption_dynamism":
        i = P.CAPTION_STYLES.index(plan["captions"]["style"]); j = max(0, min(len(P.CAPTION_STYLES) - 1, i + op["delta"]))
        plan["captions"]["style"] = P.CAPTION_STYLES[j]
        return f"Subtitles: {P.CAPTION_STYLES[i]} → {P.CAPTION_STYLES[j]}." if i != j else f"Subtitles are already at the {P.CAPTION_STYLES[i]} extreme."
    if o == "set_caption_style": plan["captions"]["style"] = op["style"]; return f"Subtitle style set to {op['style']}."
    if o == "captions_enabled": plan["captions"]["enabled"] = op["enabled"]; return "Subtitles " + ("on." if op["enabled"] else "off.")
    if o == "approve_fillers":
        n = 0
        for c in plan["cuts"]:
            if c["reason"] == "filler" and not c["enabled"]: c["enabled"] = True; c["needs_approval"] = False; n += 1
        return f"Approved removal of {n} filler(s)."
    if o == "enable_cuts":
        n = 0
        for c in plan["cuts"]:
            if c["reason"] == op.get("reason", "silence") and not c["enabled"]: c["enabled"] = True; n += 1
        return f"All detected pauses are removed ({n} re-enabled)."
    if o == "restore_all_cuts":
        n = 0
        for c in plan["cuts"]:
            if op.get("reason") in ("any", c["reason"]) and c["enabled"]: c["enabled"] = False; n += 1
        return f"Restored {n} cut(s)."
    if o == "remove_pause_near":
        s, _ = tl.to_src(op["t_out"]); best = None
        for a, b in zip(plan["transcript"], plan["transcript"][1:]):
            g0, g1 = a["end"], b["start"]
            if g1 - g0 < 0.2 or abs((g0 + g1) / 2 - s) > 4: continue
            if not (tl.is_kept(g0 + 0.05) or tl.is_kept(g1 - 0.05)): continue
            if best is None or (g1 - g0) - abs((g0 + g1) / 2 - s) * 0.1 > best[1] - best[0]: best = (g0, g1)
        if not best: return f"No pause found near {op['t_out']:.0f}s."
        plan["cuts"].append({"id": f"cut_{uuid.uuid4().hex[:6]}", "start": round(best[0] + 0.06, 3), "end": round(best[1] - 0.06, 3),
                             "reason": "manual", "enabled": True, "auto": False, "note": f"Pause near {op['t_out']:.0f}s (requested)"})
        return f"Removed a {best[1]-best[0]:.2f}s pause near {op['t_out']:.0f}s."
    if o == "cut_range":
        a, _ = tl.to_src(op["start_out"]); b, _ = tl.to_src(op["end_out"])
        plan["cuts"].append({"id": f"cut_{uuid.uuid4().hex[:6]}", "start": a, "end": b, "reason": "manual", "enabled": True, "auto": False, "note": "Requested cut"})
        return f"Cut {op['start_out']:.1f}–{op['end_out']:.1f}s."
    if o == "cut_phrase":
        i = _find_phrase(plan, op["phrase"])
        if i is None: return f"Couldn't find “{op['phrase']}” in the transcript."
        s = _sentence_of(plan, i)
        plan["cuts"].append({"id": f"cut_{uuid.uuid4().hex[:6]}", "start": round(s["start"] - 0.05, 3), "end": round(s["end"] + 0.05, 3),
                             "reason": "manual", "enabled": True, "auto": False, "note": f"Removed: “{s['text']}”"})
        return f"Removed the sentence “{s['text']}”."
    if o == "add_broll":  # a stock clip downloaded by the API into broll/<clip>.mp4
        clip = str(op.get("clip", ""))
        if not re.fullmatch(r"p[xb]\d{1,12}|ai[0-9a-f]{12}", clip): return "Unknown clip."
        op = {**op, "op": "add_photo", "photo": "", "_asset": f"broll/{clip}", "layout": op.get("layout") or ("pip" if plan.get("presenter", True) else "full_graphic")}
        o = "add_photo"
    if o == "add_photo":  # a photo the doctor uploaded (saved as photos/<id>.jpg by the API)
        pid = str(op.get("photo", ""))
        if not op.get("_asset") and not re.fullmatch(r"[0-9a-f]{12}", pid): return "Unknown photo."
        op = {**op, "op": "add_graphic", "asset": op.get("_asset") or f"photo/{pid}", "layout": op.get("layout") or ("split" if plan.get("presenter", True) else "full_graphic")}
        o = "add_graphic"
        if op.get("sentence") is not None:  # placed next to a line of the script
            sents = plan["analysis"]["sentences"]; i = int(op["sentence"])
            if not 0 <= i < len(sents): return "Unknown sentence."
            op["src_range"] = (sents[i]["start"] - 0.1, sents[i]["end"] + 0.3); op["phrase"] = ""
    if o == "delete_scene":
        n = len(plan["scenes"]); plan["scenes"] = [x for x in plan["scenes"] if x["id"] != op.get("id")]
        return "Removed." if len(plan["scenes"]) < n else "Nothing to remove."
    if o == "add_graphic":
        asset, mode = op["asset"].split("/", 1)
        s = None
        if op.get("src_range"):
            a, b = op["src_range"]
            s = next((x for x in plan["analysis"]["sentences"] if abs(x["start"] - 0.1 - a) < 0.01), None)
        elif (op.get("phrase") or "").strip():
            i = _find_phrase(plan, op["phrase"])
            if i is None: return f"Couldn't find where you discuss “{op['phrase']}”."
            s = _sentence_of(plan, i); a, b = s["start"] - 0.1, s["end"] + 0.3
        else:
            a, _ = tl.to_src(float(op.get("t_out") or 0)); b = a + 5
            if op.get("dur"): b = a + max(1.0, min(20.0, float(op["dur"])))
        if b - a < 4.5 and not op.get("dur"): b = a + 4.5
        if asset in ("photo", "broll") and not op.get("dur"): b = min(max(b, a + (4.0 if asset == "photo" else 3.5)), a + (7.0 if asset == "photo" else 6.0))
        default_layout = "split" if asset == "photo" else "pip" if asset == "broll" else {"stent": "pip", "occlusion": "split", "narrowing": "side", "sinus": "side", "gauge": "labels", "calcium": "split",
                          "scanner": "split", "coronaries": "split"}.get(mode, "split")
        plan["scenes"] = _make_room(plan["scenes"], a, b)
        plan["scenes"].append({"id": f"scn_{uuid.uuid4().hex[:6]}", "start": round(a, 3), "end": round(b, 3), "layout": op.get("layout") or default_layout,
                               "graphic": {"asset": asset, "mode": mode, "label": "Photo" if asset == "photo" else "Stock video" if asset == "broll" else op["asset"]}, "text": s["text"] if s else "",
                               "auto": False, "enabled": True, **({"caption": str(op.get("caption") or "")[:80]} if asset == "photo" else {})})
        if asset == "photo" and op.get("fill"): plan["scenes"][-1]["graphic"]["fill"] = True
        plan["scenes"].sort(key=lambda x: x["start"])
        what = ("illustration" if op.get("fill") else "photo") if asset == "photo" else "stock video" if asset == "broll" else f"{op['asset'].replace('/', ' ')} animation"
        return f"Added the {what} at {tl.to_out(a):.1f}s ({op.get('layout') or default_layout} layout)."
    if o == "remove_graphics":
        before = len(plan["scenes"])
        plan["scenes"] = [x for x in plan["scenes"] if op.get("asset") and f"{x['graphic']['asset']}/{x['graphic']['mode']}" != op["asset"]]
        return f"Removed {before - len(plan['scenes'])} graphic scene(s)."
    if o == "set_layout":
        n = 0
        for x in plan["scenes"]:
            if not op.get("asset") or op["asset"].split("/")[0] == x["graphic"]["asset"] and (op["asset"].split("/")[1] == x["graphic"]["mode"] or "/" not in op["asset"]):
                x["layout"] = op["layout"]; n += 1
        return f"Set {n} scene(s) to the {op['layout']} layout."
    if o == "face_visible":
        n = 0
        for x in plan["scenes"]:
            if x["layout"] == "full_graphic": x["layout"] = "pip"; n += 1
        return f"Switched {n} full-screen animation(s) to picture-in-picture so your face stays visible." if n else "Your face is already visible during every animation (no full-screen-only scenes)."
    if o == "set_music":
        m = plan["music"]
        if op["track"] == "off": m["enabled"] = False; return "Background music off."
        if op["track"] == "on": m["enabled"] = True; return "Background music on."
        tr = op["track"] if op["track"] != "next" else P.MUSIC_TRACKS[(P.MUSIC_TRACKS.index(m["track"]) + 1) % len(P.MUSIC_TRACKS)]
        m["track"] = tr; m["enabled"] = True; return f"Music changed to “{tr}”."
    if o == "music_volume": plan["music"]["db"] += op["delta_db"]; return f"Music level {plan['music']['db']} dB."
    if o == "zoom":
        z = plan["zoom"]; z["intensity"] = round(max(0, min(0.3, z["intensity"] + op["delta"])), 3); z["punch"] = round(max(0, min(0.3, z["punch"] + op["delta"])), 3)
        return f"Zoom intensity {z['intensity']:.2f}."
    if o == "set_title": plan["title"].update(text=op["text"], enabled=True); return f"Title set to “{op['text']}”."
    if o == "title_enabled": plan["title"]["enabled"] = op["enabled"]; return "Title " + ("on." if op["enabled"] else "off.")
    if o == "caption_mode":
        plan["captions"]["mode"] = op["mode"]
        if op.get("target"): plan["captions"]["target"] = op["target"]
        names = {"original": "in the spoken language", "translated": f"translated ({plan['captions'].get('target','en')})", "both": f"bilingual (spoken + {plan['captions'].get('target','en')})"}
        return f"Subtitles {names.get(op['mode'], op['mode'])}."
    if o == "translate_captions":
        from . import translate as TR
        cur = plan["captions"].get("translations", {})
        if cur and plan["captions"].get("target") == op["target"] and any(v.strip() for v in cur.values()):
            return "Using your existing translations."
        try:
            n = TR.fill_plan_translations(plan, op["target"]); return f"Translated {n} sentences."
        except TR.TranslationUnavailable as e:
            plan["captions"]["target"] = op["target"]; return str(e)
    if o == "set_translation":
        plan["captions"].setdefault("translations", {})[str(int(op["i"]))] = op["text"]; return "Translation updated."
    # ---- direct UI operations (same reversible path as language commands) ----
    if o == "toggle_cut":
        for c in plan["cuts"]:
            if c["id"] == op["id"]: c["enabled"] = bool(op["enabled"]); c["needs_approval"] = False; return ("Applied" if c["enabled"] else "Restored") + f" cut: {c.get('note','')}"
        return "Cut not found."
    if o == "edit_word":
        w = plan["transcript"][int(op["i"])]; old = w["text"]; w["text"] = op["text"]; w["edited"] = True
        return f"Transcript: “{old}” → “{op['text']}”."
    if o == "edit_sentence":
        return _edit_sentence(plan, int(op["i"]), op["text"])
    if o == "delete_word":
        plan["transcript"][int(op["i"])]["deleted"] = bool(op.get("deleted", True)); return "Caption word hidden."
    if o == "set_effects":
        plan.setdefault("effects", {"transitions": True, "sfx": True}).update({k: bool(v) for k, v in op.items() if k in ("transitions", "sfx")})
        return "Effects updated."
    if o == "scene_time":  # storyboard drag/resize, in edited-video (output) seconds
        x = next((y for y in plan["scenes"] if y["id"] == op.get("id")), None)
        if not x: return "Scene not found."
        a_out, b_out = float(op["start_out"]), float(op["end_out"])
        a_out = max(0.0, min(a_out, tl.duration - 0.5)); b_out = max(a_out + 0.8, min(b_out, tl.duration))
        a, _ = tl.to_src(a_out); b, _ = tl.to_src(b_out - 1e-3)
        x["start"], x["end"] = round(a, 3), round(b, 3)
        if x.get("layout") != "overlay":  # visuals on the same track can't overlap: trim the neighbours
            others = _make_room([y for y in plan["scenes"] if y is not x], x["start"], x["end"], keep_icons=True)
            plan["scenes"] = others + [x]
        plan["scenes"].sort(key=lambda y: y["start"])
        return f"Moved to {a_out:.1f}–{b_out:.1f}s."
    if o == "scene_update":
        for x in plan["scenes"]:
            if x["id"] == op["id"]:
                for k in ("layout", "enabled"):
                    if k in op: x[k] = op[k]
                if op.get("enabled") and x.get("layout") != "overlay":  # shown again: make room for it
                    plan["scenes"] = _make_room([y for y in plan["scenes"] if y is not x], x["start"], x["end"], keep_icons=True) + [x]
                    plan["scenes"].sort(key=lambda y: y["start"])
                return f"Scene updated ({x['layout']}{'' if x.get('enabled', True) else ', disabled'})."
        return "Scene not found."
    if o == "set_branding":
        plan["branding"].update({k: v for k, v in op["fields"].items() if k in plan["branding"] or k in ("logo", "image_style")}); return "Branding updated."
    return f"Unknown op {o}"


def execute_ops(plan_path, ops, label="UI edit"):
    plan = json.load(open(plan_path))
    msgs = [apply(plan, op) for op in ops]
    plan.setdefault("log", []).append({"command": label, "ops": ops, "via": "ui"})
    P.save(plan, plan_path)
    return {"ok": True, "message": " ".join(msgs), "ops": ops}


def execute(plan_path, command):
    plan = json.load(open(plan_path))
    ops = parse_rules(command)
    source = "rules"
    reply = ""
    if not ops:
        ops, reply = parse_llm(command, plan); source = "llm"
    if not ops:
        return {"ok": False, "message": reply or "I couldn't turn that into an edit. Try e.g. “make the subtitles more dynamic”, “remove the pause at 30 seconds”, “change the music to something calmer”."}
    if ops[0]["op"] == "undo":
        prev = P.undo(plan_path)
        return {"ok": bool(prev), "message": "Undone." if prev else "Nothing to undo.", "ops": ops}
    msgs = []
    for op in ops:
        try: msgs.append(apply(plan, op))
        except Exception as e: msgs.append(f"(skipped one change: {op.get('op')})")
    plan.setdefault("log", []).append({"command": command, "ops": ops, "via": source})
    P.save(plan, plan_path)
    return {"ok": True, "message": " ".join(msgs), "ops": ops, "via": source}
