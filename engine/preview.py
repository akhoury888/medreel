"""Design preview: a short example clip rendered by the real video engine, so what the doctor sees on the
New video page is exactly what the chosen look, subtitles, colours, transitions, title and music will do.
Uses a frame from the doctor's own recording when the browser sends one, else an illustrated example presenter."""
import hashlib, json, os, subprocess
import numpy as np, cv2
from . import planner as P, render as R

SAMPLE = {
    "en": (["Your heart needs oxygen every second.", "Cholesterol can narrow the arteries over time."], "What your heart needs"),
    "fr": (["Votre cœur a besoin d'oxygène à chaque seconde.", "Le cholestérol peut rétrécir les artères avec le temps."], "Ce dont votre cœur a besoin"),
    "ar": (["قلبك بحاجة للأكسجين كل ثانية.", "الكوليسترول بيضيّق الشرايين مع الوقت."], "شو بيحتاج قلبك"),
}
DUR = 6.8
FACE = [0.33, 0.17, 0.34, 0.19]


def _presenter(w=540, h=960):
    """An illustrated example presenter (no real person): clinic backdrop, white coat, stethoscope."""
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = ((np.array([226, 233, 240], np.float32) * (1 - y) + np.array([188, 200, 214], np.float32) * y) * np.ones((1, w, 1))).astype(np.uint8)
    for x0, y0, x1, y1, c in ((30, 120, 200, 420, (205, 214, 226)), (360, 90, 520, 380, (200, 210, 222)), (40, 440, 190, 470, (175, 186, 200))):
        cv2.rectangle(img, (x0, y0), (x1, y1), c, -1)
    img = cv2.GaussianBlur(img, (0, 0), 9)
    cx = w // 2
    cv2.ellipse(img, (cx, int(h * 0.98)), (int(w * 0.52), int(h * 0.36)), 0, 180, 360, (246, 248, 251), -1, cv2.LINE_AA)  # white coat
    pts = np.array([[cx - 70, int(h * .63)], [cx + 70, int(h * .63)], [cx, int(h * .80)]], np.int32)
    cv2.fillPoly(img, [pts], (52, 84, 140), cv2.LINE_AA)  # shirt
    cv2.rectangle(img, (cx - 42, int(h * .50)), (cx + 42, int(h * .64)), (222, 176, 148), -1)  # neck
    cv2.ellipse(img, (cx, int(h * .40)), (int(w * .17), int(h * .125)), 0, 0, 360, (236, 192, 164), -1, cv2.LINE_AA)  # face
    cv2.ellipse(img, (cx, int(h * .335)), (int(w * .18), int(h * .085)), 0, 180, 360, (58, 44, 38), -1, cv2.LINE_AA)  # hair
    cv2.ellipse(img, (cx, int(h * .74)), (int(w * .2), int(h * .11)), 0, 200, 340, (70, 80, 95), 6, cv2.LINE_AA)  # stethoscope
    cv2.circle(img, (cx + int(w * .19), int(h * .70)), 13, (150, 160, 175), -1, cv2.LINE_AA)
    return img  # drawn in RGB


def _source(work, frame):
    """A 6.4 s silent clip of a still frame (the engine's input)."""
    img = frame if frame is not None else _presenter()
    key = hashlib.sha1(img.tobytes()[::97]).hexdigest()[:12]
    path = os.path.join(work, f"src_{key}.mp4")
    if not os.path.exists(path):
        jp = path + ".jpg"; cv2.imwrite(jp, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", jp, "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                        "-t", f"{DUR}", "-r", "24", "-vf", "scale=540:960:force_original_aspect_ratio=increase,crop=540:960,format=yuv420p",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", path], check=True)
        os.remove(jp)
    return path, key


def make(opts, work, frame=None, branding=None):
    """opts: look, palette, captions, frame, transition, title, bg, music, lang, title_text, presenter (bool).
    -> path of a cached mp4 (360x640) showing those choices."""
    os.makedirs(work, exist_ok=True)
    src, fkey = _source(work, frame)
    lang = opts.get("lang") if opts.get("lang") in SAMPLE else "en"
    presenter = opts.get("presenter", True) is not False
    look = opts.get("look") if opts.get("look") in P.LOOKS else "clinic"
    key = hashlib.sha1(json.dumps([opts, fkey, branding or {}], sort_keys=True, default=str).encode()).hexdigest()[:16]
    out = os.path.join(work, f"pv_{key}.mp4")
    if os.path.exists(out): return out
    sents, ttl = SAMPLE[lang]
    words, sentences, t = [], [], 0.25
    for si, sen in enumerate(sents):
        toks = sen.split(); i0 = len(words); per = 2.6 / len(toks)
        for tk in toks:
            words.append({"i": len(words), "text": tk, "start": round(t, 3), "end": round(t + per * 0.85, 3)}); t += per
        sentences.append({"i0": i0, "i1": len(words) - 1, "start": words[i0]["start"], "end": words[-1]["end"], "text": sen})
        t += 0.35
    brand = {**P.DEFAULT_BRAND, **(branding or {}), "lower_third": False, "outro": False}
    plan = {
        "source": {"path": src, "width": 540, "height": 960, "duration": DUR, "fps": 24, "has_audio": True},
        "style": "modern_educational", "style_params": dict(P.STYLES["modern_educational"]), "language": lang, "presenter": presenter,
        "branding": brand, "transcript": words,
        "analysis": {"sentences": sentences, "terms": [], "qa": {"luma_mean": 120, "issues": []}},
        "face_track": [{"t": 0, "box": opts.get("face") or FACE}], "cuts": [], "emphasis": [],
        "scenes": [{"id": "pv1", "start": 3.55, "end": 6.0, "layout": "split" if presenter else "full_graphic",
                    "graphic": {"asset": "coronary", "mode": "narrowing", "label": ""}, "text": sents[1], "enabled": True}],
        "captions": {"style": "highlight", "enabled": True, "language": lang, "max_words": None, "mode": "original", "target": "en", "translations": {}},
        "zoom": {"mode": "auto", "intensity": 0, "punch": 0, "push": 0.03},
        "title": {"text": (opts.get("title_text") or ttl)[:60], "enabled": True, "style": "slide"},
        "music": {"track": "pulse", "enabled": True, "db": -22}, "audio": {"denoise": False, "loudnorm": -14, "highpass": 80},
        "effects": {"transitions": True, "sfx": True}, "grade": "natural", "output": {"width": 1080, "height": 1920, "fps": 30},
    }
    P.apply_look(plan, look, keep_brand=bool(opts.get("keep_brand")))
    plan["zoom"]["intensity"] = 0  # a still frame: no jump-cut zoom
    d = plan["design"]
    for k in P.DESIGN_FIELDS:
        if opts.get(k) in P.DESIGN_FIELDS[k]: d[k] = opts[k]
    if opts.get("captions") in P.CAPTION_LOOKS: plan["captions"]["style"] = opts["captions"]
    if opts.get("music") in P.MUSIC_TRACKS: plan["music"].update({"track": opts["music"], "enabled": True})
    elif opts.get("music") == "off": plan["music"]["enabled"] = False
    tmp = out + ".part.mp4"
    R.render(plan, tmp, os.path.join(work, "w_" + key), preview=True, log=lambda m: None, size=(360, 640, 20))
    os.replace(tmp, out)
    for f in os.listdir(work):  # cache hygiene: keep the newest ~150 clips
        pass
    files = sorted((os.path.join(work, f) for f in os.listdir(work) if f.startswith("pv_")), key=os.path.getmtime)
    for f in files[:-150]:
        try: os.remove(f)
        except OSError: pass
    import shutil; shutil.rmtree(os.path.join(work, "w_" + key), ignore_errors=True)
    return out


def face_box(img):
    """Face box [x, y, w, h] (fractions) in the doctor's frame, so the example subtitles avoid it like in the real video."""
    try:
        from .analyze import MODELS
        yp = os.path.join(MODELS, "yunet.onnx")
        if not os.path.exists(yp): return None
        h, w = img.shape[:2]
        det = cv2.FaceDetectorYN.create(yp, "", (w, h), 0.7)
        _, faces = det.detect(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if faces is None or not len(faces): return None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4]
        return [float(x / w), float(y / h), float(fw / w), float(fh / h)]
    except Exception:
        return None


def frame_from_upload(data):
    """JPEG/PNG bytes from the browser -> 540x960 RGB (cover crop)."""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None or arr.mean() < 6: return None  # black still: keep the example presenter
    h, w = arr.shape[:2]; k = max(540 / w, 960 / h)
    arr = cv2.resize(arr, (int(w * k + 1), int(h * k + 1)), interpolation=cv2.INTER_AREA)
    y0, x0 = (arr.shape[0] - 960) // 2, (arr.shape[1] - 540) // 2
    return cv2.cvtColor(arr[y0:y0 + 960, x0:x0 + 540], cv2.COLOR_BGR2RGB)
