"""AI medical illustrations: Claude writes the image brief from what the doctor says, FLUX.2 (fal.ai) draws it.

Keys: FAL_KEY (fal.ai) for the images; ANTHROPIC_API_KEY for the briefs (without it, briefs come from the
concept vocabulary). Images are saved into the project's photo library (photos/<id>.jpg + .json with the brief)
and placed like the doctor's own photos, full-bleed with a slow camera move. Every image is reviewable,
removable and regenerable in the studio — AI anatomy can be wrong, the doctor has the last word."""
import base64, concurrent.futures as cf, json, logging, os, re, urllib.error, urllib.request, uuid

log = logging.getLogger("medreel.imagegen")
FAL_MODEL = os.environ.get("MEDREEL_IMAGE_MODEL", "fal-ai/flux-2-pro")
SIZE = {"width": 720, "height": 1280}  # 9:16, under 1 MP -> one price unit; upscaled to 1080x1920 at render
STYLES = {
    "3d": "Glossy 3D medical illustration, high-end medical animation still, cinematic studio lighting, deep navy background, "
          "soft {accent} rim light, clean composition with the subject centred, anatomically accurate, ultra detailed, no text, no labels, no watermark",
    "flat": "Clean modern flat vector medical illustration, infographic style, soft gradients, navy and {accent} colour palette, "
            "simple shapes, centred subject, anatomically accurate, no text, no labels, no watermark",
    "photo": "Realistic editorial photograph, natural soft light, shallow depth of field, modern bright clinic or everyday setting, "
             "authentic, no text, no logos, no watermark, no identifiable patient faces",
}
STYLE_NAMES = {"3d": "3D medical", "flat": "Flat illustration", "photo": "Photo-realistic"}


def available():
    return bool(os.environ.get("FAL_KEY"))


def _accent_name(hexcol):
    try:
        r, g, b = (int(hexcol.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return "electric blue"
    if r > 180 and g < 120: return "crimson red"
    if g > 150 and r < 120 and b < 150: return "teal green"
    if r > 200 and g > 150 and b < 100: return "warm gold"
    return "electric blue"


def style_prompt(style, accent="#2EA0FF"):
    return STYLES.get(style, STYLES["3d"]).format(accent=_accent_name(accent))


# ------------------------------------------------------------------ image briefs
def _claude(prompt, max_tokens=2000):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key: return None
    body = json.dumps({"model": os.environ.get("MEDREEL_LLM", "claude-sonnet-5-5"), "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body,
                                 {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=90).read())["content"][0]["text"]


BRIEF_RULES = ("Rules for every brief: describe ONE clear visual subject in English (20-45 words); medically and anatomically accurate; "
               "no text, letters, numbers or labels in the image; no identifiable real people, no patients' faces, no gore, no needles "
               "piercing skin in close-up, nothing frightening; prefer organs, cells, devices, healthy everyday scenes and symbolic "
               "visuals; never contradict what the doctor said.")


def briefs_for(sentences, specialty="", want=5, exclude=()):
    """Claude picks the sentences that benefit most from an illustration and writes a brief for each.
    sentences: [(index, text)]. -> [{"i", "brief", "caption"}] (caption in the video's language, 1-4 words)."""
    cands = [(i, t) for i, t in sentences if i not in set(exclude) and len(t.split()) >= 3]
    if not cands or want <= 0: return []
    txt = _claude(
        f"You are the art director of a medical social-media agency. A {specialty or 'physician'} says these numbered sentences "
        f"(they may be in Lebanese Arabic, French or English):\n" + "\n".join(f"{i}: {t}" for i, t in cands) +
        f"\n\nChoose up to {want} sentences that would gain the most from an illustration (spread through the video, skip greetings "
        f"and sign-offs) and write an image brief for each. {BRIEF_RULES}\nReturn ONLY JSON: "
        '[{"i": <sentence number>, "brief": "...", "caption": "<1-4 words in the sentence language>"}]')
    if txt is None: return None  # no Claude key: caller falls back to the concept vocabulary
    try:
        out = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
    except ValueError:
        return []
    ok = {i for i, _ in cands}
    return [{"i": int(o["i"]), "brief": str(o["brief"])[:600], "caption": str(o.get("caption", ""))[:40]}
            for o in out if isinstance(o, dict) and int(o.get("i", -1)) in ok and o.get("brief")][:want]


def brief_for_sentence(text, specialty=""):
    txt = _claude(f"A {specialty or 'physician'} says: \"{text}\"\nWrite one image brief that illustrates this for a patient-education "
                  f"video. {BRIEF_RULES}\nReturn ONLY the brief.", max_tokens=300)
    return (txt or "").strip().strip('"')[:600] or None


def concept_brief(key):
    from . import concepts as CO
    subj = {"lungs": "a pair of healthy human lungs with the bronchial tree", "asthma": "inflamed narrowed airways inside the lungs during an asthma attack",
            "inhaler": "a blue asthma inhaler with a fine mist", "smoking": "a pair of lungs darkened by cigarette smoke next to a burning cigarette",
            "shisha": "a hookah with swirling smoke drifting toward a pair of lungs", "quit_smoking": "a broken cigarette next to clean healthy lungs",
            "copd": "damaged lungs with enlarged air sacs (emphysema)", "short_breath": "lungs with constricted airways and a faint struggling airflow",
            "oxygen": "oxygen molecules flowing into healthy lungs", "saturation": "a pulse oximeter on a fingertip with a glowing pulse wave",
            "heart": "a healthy human heart with its coronary arteries", "blood_pressure": "a blood pressure cuff and a pulsing artery",
            "cholesterol": "a cross-section of an artery with yellow cholesterol plaque", "diabetes": "glucose molecules in a blood vessel next to a glucose meter",
            "kidney": "a pair of healthy human kidneys", "liver": "a healthy human liver", "stomach": "the human stomach", "brain": "the human brain with glowing neurons",
            "exercise": "a person jogging at sunrise, seen from behind", "walking": "a person walking in a green park, seen from behind",
            "diet": "a colourful plate of vegetables, fruit and whole grains", "salt": "a salt shaker with a red cross-out symbol shape", "sleep": "a calm bedroom at night",
            "medication": "a few white tablets and a blister pack", "vaccine": "a vaccine vial with antibodies protecting a cell", "virus": "virus particles near a cell",
            "bones": "a healthy human skeleton detail of the hip and spine", "joints": "a healthy human knee joint", "pregnancy": "a pregnant woman's silhouette holding her belly"}
    return subj.get(key) or f"a medical illustration of {CO.label(key, 'en').lower()}"


# ------------------------------------------------------------------ generation
def generate(brief, project_dir, style="3d", accent="#2EA0FF", caption="", seed=None):
    """-> photo id (photos/<id>.jpg) or raises."""
    payload = {"prompt": f"{brief}. {style_prompt(style, accent)}", "image_size": SIZE, "output_format": "jpeg",
               "enable_safety_checker": True, "safety_tolerance": "2"}
    if seed is not None: payload["seed"] = int(seed)
    hdr = {"Authorization": f"Key {os.environ['FAL_KEY']}", "Content-Type": "application/json"}
    try:
        res = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://fal.run/{FAL_MODEL}", json.dumps(payload).encode(), hdr),
                                                timeout=180).read())
    except urllib.error.HTTPError as e:
        if e.code == 422:  # model without custom sizes
            payload["image_size"] = "portrait_16_9"
            res = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://fal.run/{FAL_MODEL}", json.dumps(payload).encode(), hdr),
                                                    timeout=180).read())
        else:
            raise RuntimeError(f"image service refused ({e.code})")
    url = (res.get("images") or [{}])[0].get("url")
    if not url: raise RuntimeError("no image returned")
    if url.startswith("data:"):
        data = base64.b64decode(url.split(",", 1)[1])
    else:
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "MedReel/1.0"}), timeout=60).read()
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGB")
    d = os.path.join(project_dir, "photos"); os.makedirs(d, exist_ok=True)
    pid = uuid.uuid4().hex[:12]
    im.save(os.path.join(d, pid + ".jpg"), "JPEG", quality=92)
    json.dump({"ai": True, "brief": brief, "style": style, "model": FAL_MODEL, "caption": caption, "seed": res.get("seed")},
              open(os.path.join(d, pid + ".json"), "w"), ensure_ascii=False)
    return pid


def auto(plan, project_dir, want=None, log_fn=None):
    """Illustrate the best moments of a new video. Procedural medical animations are kept; an AI image replaces the
    icon card (or fills the gap) at its sentence. Best effort — failures leave the plan as it was."""
    if not available(): return 0
    from . import commands as C, concepts as CO
    presenter = plan.get("presenter", True)
    dur = plan["source"]["duration"]
    want = want if want is not None else int(os.environ.get("MEDREEL_AI_IMAGES", "0") or 0) or max(2, min(6, int(dur / 10)))
    sents = plan["analysis"]["sentences"]
    words = plan["transcript"]
    busy = [(s["start"], s["end"]) for s in plan["scenes"] if s["graphic"]["asset"] not in ("icons",)]
    free = [(i, s["text"]) for i, s in enumerate(sents)
            if not any(a < s["end"] - 0.3 and b > s["start"] + 0.3 for a, b in busy) and s["start"] > (4.0 if presenter else 3.0)]
    picks = None
    try:
        picks = briefs_for(free, plan["branding"].get("specialty", ""), want)
    except Exception as e:
        (log_fn or log.warning)(f"image briefs unavailable ({e})")
    if picks is None:  # no Claude key: concept vocabulary
        picks = []
        for i, t in free:
            hits = CO.find(words[sents[i]["i0"]: sents[i]["i1"] + 1])
            if hits: picks.append({"i": i, "brief": concept_brief(hits[0][1]), "caption": ""})
        step = max(1, len(picks) // max(1, want)); picks = picks[::step][:want]
    if not picks: return 0
    style = plan["branding"].get("image_style", "3d"); accent = plan["branding"].get("primary", "#2EA0FF")
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(generate, p["brief"], project_dir, style, accent, p.get("caption", "")): p for p in picks}
        done = []
        for f in cf.as_completed(futs):
            try: done.append((futs[f], f.result()))
            except Exception as e: (log_fn or log.warning)(f"image failed ({e})")
    n = 0
    for p, pid in sorted(done, key=lambda x: x[0]["i"]):
        s = sents[p["i"]]
        plan["scenes"] = [x for x in plan["scenes"] if not (x["graphic"]["asset"] == "icons" and x["start"] < s["end"] and x["end"] > s["start"])]
        msg = C.apply(plan, {"op": "add_photo", "photo": pid, "sentence": p["i"], "fill": True,
                             "layout": "split" if presenter else "full_graphic", "caption": ""})
        if msg.startswith("Added"): n += 1
    return n


# ------------------------------------------------------------------ animation (image -> 5 s video)
ANIM_MODEL = os.environ.get("MEDREEL_ANIM_MODEL", "fal-ai/kling-video/v2.6/pro/image-to-video")
MOTION_RULES = ("Slow, smooth, cinematic medical-animation motion; the camera drifts or pushes in gently; the subject moves subtly and "
                "naturally (e.g. lungs breathing, blood flowing, particles drifting); anatomy keeps its shape; no morphing, no new objects, "
                "no text, no people appearing, no cuts.")


def _fal_queue(model, payload, timeout=420):
    """fal queue API: submit, poll, fetch — long video jobs don't hold one HTTP request open."""
    import time
    hdr = {"Authorization": f"Key {os.environ['FAL_KEY']}", "Content-Type": "application/json"}
    try:
        sub = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://queue.fal.run/{model}", json.dumps(payload).encode(), hdr),
                                                timeout=60).read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"animation service refused ({e.code})")
    status_url = sub.get("status_url") or f"https://queue.fal.run/{model}/requests/{sub['request_id']}/status"
    result_url = sub.get("response_url") or f"https://queue.fal.run/{model}/requests/{sub['request_id']}"
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = json.loads(urllib.request.urlopen(urllib.request.Request(status_url, headers=hdr), timeout=30).read())
        if st.get("status") == "COMPLETED": break
        if st.get("status") in ("FAILED", "ERROR", "CANCELLED"): raise RuntimeError("animation failed")
        time.sleep(4)
    else:
        raise RuntimeError("animation took too long")
    return json.loads(urllib.request.urlopen(urllib.request.Request(result_url, headers=hdr), timeout=60).read())


def motion_prompt(brief):
    txt = None
    try:
        txt = _claude(f"Write one sentence (max 40 words) describing how this medical illustration should come alive in a 5-second "
                      f"clip. Illustration: \"{brief}\". {MOTION_RULES} Return ONLY the sentence.", max_tokens=150)
    except Exception:
        pass
    return ((txt or "").strip().strip('"') or "Gentle cinematic push-in, subtle natural motion of the subject.") + " " + MOTION_RULES


def animate(photo_id, project_dir):
    """Turn an AI illustration (photos/<id>.jpg) into a 5 s clip -> clip id 'ai<id>' in broll/ (same format as stock clips)."""
    import subprocess
    src = os.path.join(project_dir, "photos", photo_id + ".jpg")
    meta_p = os.path.join(project_dir, "photos", photo_id + ".json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    data_uri = "data:image/jpeg;base64," + base64.b64encode(open(src, "rb").read()).decode()
    res = _fal_queue(ANIM_MODEL, {"prompt": motion_prompt(meta.get("brief", "a medical illustration")), "start_image_url": data_uri,
                                  "duration": "5", "generate_audio": False, "negative_prompt": "blur, distortion, morphing anatomy, text, watermark, low quality"})
    url = (res.get("video") or {}).get("url")
    if not url: raise RuntimeError("no video returned")
    d = os.path.join(project_dir, "broll"); os.makedirs(d, exist_ok=True)
    clip = "ai" + photo_id
    raw = os.path.join(d, clip + ".src.mp4")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "MedReel/1.0"}), timeout=120) as r, open(raw, "wb") as fo:
        fo.write(r.read(200 * 1024 * 1024))
    out = os.path.join(d, clip + ".mp4")
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30"
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-an", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
                        "-pix_fmt", "yuv420p", out], capture_output=True)
    os.remove(raw)
    if r.returncode or not os.path.exists(out): raise RuntimeError("could not read the animation")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", out, "-frames:v", "1", "-vf", "scale=240:-2", os.path.join(d, clip + ".jpg")])
    json.dump({"source": "ai-animation", "photo": photo_id, "model": ANIM_MODEL}, open(os.path.join(d, clip + ".json"), "w"))
    return clip


def animate_scene(plan, scene_id, project_dir):
    """Swap an AI-illustration scene for its animated version (keeps timing and layout)."""
    sc = next((x for x in plan["scenes"] if x["id"] == scene_id), None)
    if not sc or sc["graphic"]["asset"] != "photo" or not sc["graphic"].get("fill"): raise ValueError("Only AI illustrations can be animated.")
    clip = animate(sc["graphic"]["mode"], project_dir)
    sc["graphic"] = {"asset": "broll", "mode": clip, "label": "Animated illustration", "photo": sc["graphic"]["mode"], "animated": True}
    return clip


def auto_animate(plan, project_dir, n=None, log_fn=None):
    """Animate the first few AI illustrations of a new video (cost-capped: MEDREEL_AI_ANIMATE, default 2)."""
    n = int(os.environ.get("MEDREEL_AI_ANIMATE", "2")) if n is None else n
    targets = [x["id"] for x in sorted(plan["scenes"], key=lambda x: x["start"]) if x["graphic"]["asset"] == "photo" and x["graphic"].get("fill")][:n]
    if not targets: return 0
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(animate, next(x for x in plan["scenes"] if x["id"] == sid)["graphic"]["mode"], project_dir): sid for sid in targets}
        for f in cf.as_completed(futs):
            try:
                clip = f.result(); sc = next(x for x in plan["scenes"] if x["id"] == futs[f])
                sc["graphic"] = {"asset": "broll", "mode": clip, "label": "Animated illustration", "photo": sc["graphic"]["mode"], "animated": True}
                ok += 1
            except Exception as e:
                (log_fn or log.warning)(f"animation failed ({e})")
    return ok
