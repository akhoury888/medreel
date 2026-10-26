"""Stock B-roll from Pexels or Pixabay (both free for commercial use, no attribution required in the video).

Needs PEXELS_API_KEY (pexels.com/api) or PIXABAY_API_KEY (pixabay.com/api/docs). Clips are downloaded once per
project (never hot-linked), trimmed to <= 8 s, cropped to 1080x1920 at 30 fps without audio (the doctor's voice
continues over them), stored as <project>/broll/<clip>.mp4 + .jpg poster, and used as a 'broll/<clip>' scene.
Clip ids carry the source: 'px<id>' = Pexels, 'pb<id>' = Pixabay."""
import json, logging, os, re, subprocess, urllib.parse, urllib.request
from . import concepts as CO

log = logging.getLogger("medreel.broll")
API = "https://api.pexels.com/videos"
PIXABAY = "https://pixabay.com/api/videos/"
CLIP_RE = re.compile(r"p[xb]\d{1,12}")
MUST = {"lungs": ["lung", "breath", "chest"], "breathing": ["breath", "inhal", "exhal"], "short_breath": ["breath", "tired", "stairs", "exhaust"],
        "asthma": ["asthma", "inhaler"], "inhaler": ["inhaler", "asthma"], "cough": ["cough", "sick", "flu", "cold"],
        "smoking": ["smok", "cigarette", "tobacco"], "shisha": ["hookah", "shisha", "narg", "smok"], "quit_smoking": ["cigarette", "smok", "tobacco"],
        "oxygen": ["oxygen", "mask", "hospital", "patient"], "saturation": ["oximeter", "pulse", "finger", "oxygen"], "sleep_apnea": ["snor", "sleep", "bed"],
        "xray": ["xray", "x-ray", "radiolog", "radiograph", "skeleton", "chest"], "allergy": ["allerg", "pollen", "sneez", "tissue", "flower"],
        "virus": ["virus", "covid", "corona", "bacteri", "microb", "cell"], "vaccine": ["vaccin", "syringe", "inject", "needle"],
        "blood_pressure": ["blood pressure", "pressure", "tonometer", "sphygmo", "cuff"], "medication": ["pill", "tablet", "medic", "drug", "capsule", "pharma"],
        "blood_test": ["blood", "test tube", "laborator", "sample"], "lab": ["laborator", "test tube", "microscop", "scien"],
        "ultrasound": ["ultrasound", "sonogra", "echo", "scan"], "surgery": ["surg", "operat", "hospital"], "hospital": ["hospital", "clinic", "medical", "doctor", "nurse"],
        "emergency": ["ambulance", "emergenc", "siren", "paramedic"], "checkup": ["doctor", "patient", "medical", "stethoscope", "clinic"],
        "headache": ["headache", "migraine", "pain", "stress"], "fever": ["fever", "thermometer", "sick", "temperature"], "fatigue": ["tired", "sleep", "fatigue", "exhaust"],
        "exercise": ["exercis", "sport", "fitness", "run", "jog", "gym", "workout"], "walking": ["walk", "hik", "stroll"], "swimming": ["swim", "pool"],
        "cycling": ["cycl", "bike", "bicycle"], "yoga": ["yoga", "stretch", "meditat"], "weight": ["scale", "weight", "diet", "fitness"],
        "diet": ["vegetable", "fruit", "salad", "healthy", "food"], "fast_food": ["burger", "fries", "fast food", "junk", "pizza"], "water": ["water", "drink", "glass", "hydrat"],
        "sleep": ["sleep", "bed", "rest"], "stress": ["stress", "anxi", "work", "tired", "worr"], "diabetes": ["diabet", "glucose", "insulin", "blood sugar"],
        "back": ["back", "spine", "pain", "posture"], "skin": ["skin", "sunscreen", "cream", "derma"], "pregnancy": ["pregnan", "belly", "maternity"],
        "elderly": ["elderly", "senior", "old", "grandp", "grandm", "retire"]}
ANIMALS = ("dog", "cat", "puppy", "kitten", "pet", "animal", "horse", "bird", "cow", "sheep", "wildlife", "zoo")


def _relevant(r, key):
    tags = (r.get("tags") or "").lower().replace("-", "")
    if any(re.search(rf"\b{a}s?\b", tags) for a in ANIMALS): return False
    must = MUST.get(key)
    if not must or not tags: return True  # Pexels gives no tags: trust its (better) relevance
    return any(m.replace("-", "") in tags for m in must)


def provider():
    return "pexels" if os.environ.get("PEXELS_API_KEY") else "pixabay" if os.environ.get("PIXABAY_API_KEY") else None


def available():
    return provider() is not None


def credit():
    return {"pexels": ("Pexels", "https://www.pexels.com"), "pixabay": ("Pixabay", "https://pixabay.com")}.get(provider())


def _get(url, timeout=20):
    hdr = {"User-Agent": "MedReel/1.0"}
    if "pexels.com" in url: hdr["Authorization"] = os.environ["PEXELS_API_KEY"]
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=timeout).read())


def _brief(v):  # Pexels video -> common shape
    return {"id": f"px{v['id']}", "image": v.get("image"), "duration": v.get("duration"), "user": (v.get("user") or {}).get("name"),
            "url": v.get("url"), "_files": [{"link": f.get("link"), "width": f.get("width"), "height": f.get("height")}
                                             for f in v.get("video_files", []) if (f.get("file_type") or "").endswith("mp4")]}


def _brief_pb(h):  # Pixabay hit -> common shape
    vs = h.get("videos") or {}
    thumb = next((vs[k].get("thumbnail") for k in ("small", "medium", "tiny", "large") if (vs.get(k) or {}).get("thumbnail")), None)
    return {"id": f"pb{h['id']}", "image": thumb, "duration": h.get("duration"), "user": h.get("user"), "url": h.get("pageURL"), "tags": h.get("tags", ""),
            "_files": [{"link": f.get("url"), "width": f.get("width"), "height": f.get("height")} for f in vs.values() if f and f.get("url")]}


def _pb_url(**params):
    return PIXABAY + "?" + urllib.parse.urlencode({"key": os.environ["PIXABAY_API_KEY"], "safesearch": "true", **params})


def search(query, per_page=8, orientation="portrait"):
    if provider() == "pixabay":
        hits = _get(_pb_url(q=query[:100], per_page=max(3, per_page), video_type="film")).get("hits", [])
        out = [_brief_pb(h) for h in hits if 3 <= (h.get("duration") or 0) <= 60]
        # portrait/square clips first (less cropping on a 9:16 screen)
        return sorted(out, key=lambda r: -max((f["height"] or 0) / max(1, f["width"] or 1) for f in r["_files"]) if r["_files"] else 0)
    q = urllib.parse.urlencode({"query": query[:80], "per_page": per_page, "orientation": orientation, "size": "medium"})
    vids = _get(f"{API}/search?{q}").get("videos", [])
    return [_brief(v) for v in vids if 3 <= (v.get("duration") or 0) <= 60]


def _lookup(clip_id):
    if clip_id.startswith("pb"):
        hits = _get(_pb_url(id=int(clip_id[2:]))).get("hits", [])
        return _brief_pb(hits[0]) if hits else None
    return _brief(_get(f"{API}/videos/{int(clip_id[2:])}"))


def _pick_file(files):
    """Smallest file that is still >= 1280 px on its long side; else the largest."""
    mp4 = [f for f in files if f.get("link")]
    good = [f for f in mp4 if max(f.get("width") or 0, f.get("height") or 0) >= 1280]
    if good: return min(good, key=lambda f: (f.get("width") or 0) * (f.get("height") or 0))
    return max(mp4, key=lambda f: (f.get("width") or 0) * (f.get("height") or 0)) if mp4 else None


def fetch(video, project_dir, max_s=8.0):
    """video: a search result or a clip id ('px123' / 'pb123'). -> clip id or None."""
    if not isinstance(video, dict):
        if not CLIP_RE.fullmatch(str(video)): return None
        video = _lookup(str(video))
        if not video: return None
    clip = video["id"]
    if not CLIP_RE.fullmatch(clip): return None
    d = os.path.join(project_dir, "broll"); os.makedirs(d, exist_ok=True)
    out = os.path.join(d, clip + ".mp4")
    if os.path.exists(out): return clip
    f = _pick_file(video.get("_files") or [])
    if not f: return None
    raw = os.path.join(d, clip + ".src.mp4")
    req = urllib.request.Request(f["link"], headers={"User-Agent": "MedReel/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(raw, "wb") as fo:
        n = 0
        while True:
            b = r.read(1 << 20)
            if not b: break
            n += len(b); fo.write(b)
            if n > 150 * 1024 * 1024: raise RuntimeError("clip too large")
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,eq=saturation=1.05"
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-t", f"{max_s:.1f}", "-an", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "21", "-pix_fmt", "yuv420p", out], capture_output=True)
    os.remove(raw)
    if r.returncode or not os.path.exists(out): return None
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", out, "-frames:v", "1", "-vf", "scale=240:-2", os.path.join(d, clip + ".jpg")])
    json.dump({"source": "pixabay" if clip.startswith("pb") else "pexels", "url": video.get("url"), "user": video.get("user")}, open(os.path.join(d, clip + ".json"), "w"))
    return clip


def auto(plan, project_dir, max_per_min=4, log_fn=None):
    """Turn some icon cards into real stock footage of the same concept (spaced >= 10 s apart).
    Best effort: any network/API problem leaves the icon cards as they are."""
    if not available(): return 0
    presenter = plan.get("presenter", True)
    dur = plan["source"]["duration"]; limit = max(1, int(dur / 60 * max_per_min)); used, last, done = set(), -99.0, 0
    for sc in sorted(plan["scenes"], key=lambda x: x["start"]):
        if done >= limit: break
        g = sc["graphic"]
        if g["asset"] != "icons" or sc["start"] - last < 10.0: continue
        key = next((it["key"] for it in g.get("items", []) if CO.stock_query(it["key"]) and it["key"] not in used), None)
        if not key: continue
        try:
            res = [r for r in search(CO.stock_query(key), per_page=15) if _relevant(r, key)]
            clip = next((c for c in (fetch(r, project_dir) for r in res[:2]) if c), None)
        except Exception as e:
            (log_fn or log.warning)(f"stock video unavailable ({e}); keeping icon cards"); return done
        if not clip: continue
        sc["graphic"] = {"asset": "broll", "mode": clip, "label": CO.label(key, "en"), "concept": key}
        sc["layout"] = "pip" if presenter else "full_graphic"
        nxt = min([x["start"] for x in plan["scenes"] if x["start"] > sc["start"] + 0.01] + [dur])
        sc["end"] = round(min(max(sc["end"], sc["start"] + 3.5), nxt - 0.3, sc["start"] + 6.0), 3)
        used.add(key); last = sc["start"]; done += 1
    return done
