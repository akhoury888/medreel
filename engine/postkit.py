"""Everything a doctor needs to post the video: the caption (post text + hashtags) and a cover image.

caption(plan, lang)  -> {"text", "hashtags": [...], "by": "claude"|"template"}
covers(plan, project_dir) -> {"cover": path (1080x1920), "square": path (1080x1080)}
"""
import json, os, re, subprocess
import numpy as np, cv2
from . import planner as P, knowledge as K

LANG_NAME = {"ar": "Lebanese Arabic (natural, as Lebanese doctors write on Instagram)", "en": "English", "fr": "French"}
DISCLAIMER = {"ar": "⚠️ محتوى توعوي — لا يغني عن استشارة طبيبك.", "en": "⚠️ Educational content — not individual medical advice.",
              "fr": "⚠️ Contenu éducatif — ne remplace pas un avis médical personnalisé."}
FOLLOW = {"ar": "تابعوني لمعلومات أكتر عن صحتكن 👨‍⚕️", "en": "Follow for more health tips 👨‍⚕️", "fr": "Abonnez-vous pour plus de conseils santé 👨‍⚕️"}


def _script(plan, limit=2500):
    return " ".join(s["text"] for s in plan["analysis"]["sentences"])[:limit]


def caption(plan, lang="en", platform="instagram"):
    """Ready-to-post text: a hook line, a short summary, a call to action, the disclaimer and hashtags.
    Written by Claude when a key is set; otherwise a clean template from the video's own words."""
    lang = lang if lang in LANG_NAME else "en"
    b = plan.get("branding", {})
    try:
        from .imagegen import _claude
        txt = _claude(
            f"You write social-media captions for a doctor's short video ({platform}).\n"
            f"Doctor: {b.get('name', '')} — {b.get('specialty', '')}{(' — ' + b['clinic']) if b.get('clinic') else ''}.\n"
            f"What the doctor says in the video (may be another language):\n\"{_script(plan)}\"\n\n"
            f"Write the caption in {LANG_NAME[lang]}. Rules: first line = a short hook that makes people stop scrolling; then 2-4 short lines "
            "summarising the key message (a few emojis are fine, no exaggeration, no promises of cure, nothing the doctor did not say); "
            "then one call to action (follow / save / share / book a consultation); keep it under 900 characters. "
            "Then 8-12 relevant hashtags (mix of the specialty, the topic, and general health; in the caption's language and English). "
            'Return ONLY JSON: {"text": "...", "hashtags": ["#...", "..."]}', 1200)
        if txt:
            obj = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
            tags = [("#" + h.lstrip("#").replace(" ", "")) for h in obj.get("hashtags", []) if str(h).strip()][:14]
            text = str(obj.get("text", "")).strip()
            if text:
                if DISCLAIMER[lang] not in text: text += "\n\n" + DISCLAIMER[lang]
                return {"text": text, "hashtags": tags, "by": "claude", "lang": lang}
    except Exception:
        pass
    # template fallback: the title as hook + the first sentences + CTA + disclaimer
    sents = [s["text"].strip() for s in plan["analysis"]["sentences"] if len(s["text"].split()) >= 3]
    hook = (plan.get("title") or {}).get("text") or (sents[0] if sents else "")
    body = " ".join(sents[1:3]) if len(sents) > 1 else ""
    terms = []
    for t in K.detect_terms(plan.get("transcript", [])):
        lab = re.sub(r"\s+", "", t["label"])
        if lab and lab not in terms: terms.append(lab)
    spec = re.sub(r"[^\w]", "", (b.get("specialty") or "health").split()[0]) if b.get("specialty") else "health"
    tags = ["#" + x for x in (terms[:5] + [spec, "health", "doctor", "medicaltips"])][:10]
    text = "\n\n".join(x for x in (hook, body, FOLLOW[lang], DISCLAIMER[lang]) if x)
    return {"text": text, "hashtags": tags, "by": "template", "lang": lang}


# ------------------------------------------------------------------ cover images
def _frame(path, t):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0, t):.2f}", "-i", path, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                       capture_output=True)
    if not r.stdout: return None
    im = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None


def _cover_crop(img, W, H, cx=0.5, cy=0.4):
    """Scale to cover W x H, keeping the point (cx, cy) (fractions) as central as possible."""
    h, w = img.shape[:2]; k = max(W / w, H / h)
    big = cv2.resize(img, (int(w * k + 1), int(h * k + 1)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    x0 = int(min(max(0, cx * big.shape[1] - W / 2), big.shape[1] - W)); y0 = int(min(max(0, cy * big.shape[0] - H * 0.42), big.shape[0] - H))
    return big[y0:y0 + H, x0:x0 + W].copy()


def _base(plan, project_dir):
    """The best still: the doctor's face early in the video, or (voice only) a frame of the first visual."""
    from . import render as R
    tl = P.Timeline(P.keep_segments(plan))
    if plan.get("presenter", True):
        sents = plan["analysis"]["sentences"]
        t_src = (sents[0]["start"] + sents[0]["end"]) / 2 if sents else tl.to_src(1.0)[0]
        img = _frame(plan["source"]["path"], t_src)
        face = R._face_at(plan.get("face_track"), t_src) if img is not None else None
        if img is not None: return img, ((face[0] + face[2] / 2), (face[1] + face[3] / 2)) if face else (0.5, 0.35), True
    sc = next((x for x in plan["scenes"] if x.get("enabled", True) and x.get("layout") != "overlay" and x["graphic"]["asset"] != "icons"), None)
    if sc is not None:  # voice only: the first visual itself, drawn clean (no subtitles baked in)
        try:
            from . import graphics as G, graphics_pro as GP, motion as MO
            from .draw import hex2rgb
            base_dir = os.path.dirname(os.path.abspath(plan["source"]["path"]))
            GP.PHOTO_DIR = os.path.join(base_dir, "photos"); GP.BROLL_DIR = os.path.join(base_dir, "broll"); GP._READERS.clear()
            MO.GEN_DIR = os.path.join(base_dir, "anims"); G.UI_SCALE = 1.0
            g = sc["graphic"]; D = max(1.0, sc["end"] - sc["start"])
            kw = {"title": g["title"]} if g.get("title") else {}
            if g.get("fill"): kw["fill"] = True
            img = G.ASSETS[g["asset"]](1080, 1920, min(D * 0.6, 3.5), D, mode=g["mode"], accent=hex2rgb(P.colors(plan)[0]),
                                       lang=plan.get("language", "en"), **kw)
            if img is not None: return np.ascontiguousarray(img[..., :3]), (0.5, 0.42), False
        except Exception:
            pass
    for name in ("final.mp4", "preview.mp4"):
        p = os.path.join(project_dir, "renders", name)
        if os.path.exists(p):
            img = _frame(p, 0.6)
            if img is not None: return img, (0.5, 0.45), False
    return None, (0.5, 0.4), False


def _compose(base, centre, presenter, plan, W, H):
    from .draw import hex2rgb, blit, scale_rgba
    from . import render as R
    acc_hex, sec_hex = P.colors(plan); accent = hex2rgb(acc_hex)
    img = _cover_crop(base, W, H, *centre).astype(np.float32)
    img = cv2.convertScaleAbs(img, alpha=1.06, beta=4).astype(np.float32)  # a touch of punch
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    shade = np.clip((y - 0.45) / 0.55, 0, 1) ** 1.3 * 0.78 + np.clip((0.12 - y) / 0.12, 0, 1) * 0.35  # readable text at the bottom
    img = (img * (1 - shade)).astype(np.uint8)
    s = W / 1080
    title = (plan.get("title") or {}).get("text") or (plan["analysis"]["sentences"][0]["text"] if plan["analysis"]["sentences"] else "")
    kind = plan.get("design", {}).get("title", "classic")
    sprites = R._title_lines(title, s * (1.25 if H > W else 1.05), W, accent, "boxed" if kind == "boxed" else ("big" if kind == "big" else "classic"))
    total = sum(sp.shape[0] for sp in sprites) + 14 * s * (len(sprites) - 1)
    yy = H * (0.88 if H > W else 0.90) - total
    for sp in sprites:
        if sp.shape[1] > W * 0.92: sp = scale_rgba(sp, W * 0.92 / sp.shape[1])
        blit(img, sp, (W - sp.shape[1]) / 2, yy); yy += sp.shape[0] + 14 * s
    b = plan.get("branding", {})
    logo = R._logo(b, int(130 * s), accent, hex2rgb(sec_hex))
    blit(img, scale_rgba(logo, (110 * s) / logo.shape[1]), 44 * s, 44 * s, 0.95)
    if b.get("name"):
        from .draw import text_sprite, fit_font
        nm = text_sprite(b["name"], fit_font(b["name"], "Poppins", 600, int(34 * s), W * 0.6), (255, 255, 255), shadow=4)
        blit(img, nm, 44 * s + 124 * s, 44 * s + (110 * s - nm.shape[0]) / 2, 0.95)
    return img


def covers(plan, project_dir):
    base, centre, presenter = _base(plan, project_dir)
    if base is None: raise RuntimeError("no frame available yet — make the preview first")
    out = {}
    for key, (W, H) in (("cover", (1080, 1920)), ("square", (1080, 1080))):
        im = _compose(base, centre, presenter, plan, W, H)
        p = os.path.join(project_dir, "renders", f"{key}.jpg"); os.makedirs(os.path.dirname(p), exist_ok=True)
        cv2.imwrite(p, cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]); out[key] = p
    return out
