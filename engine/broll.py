"""Stock B-roll from Pexels (free for commercial use, no attribution required in the video).

Needs PEXELS_API_KEY (free at pexels.com/api). Clips are downloaded once per project, trimmed to
<= 8 s, cropped to 1080x1920 at 30 fps without audio (the doctor's voice continues over them),
stored as <project>/broll/<clip>.mp4 + .jpg poster, and used as a 'broll/<clip>' scene."""
import json, logging, os, re, subprocess, urllib.parse, urllib.request
from . import concepts as CO

log = logging.getLogger("medreel.broll")
API = "https://api.pexels.com/videos"


def available():
    return bool(os.environ.get("PEXELS_API_KEY"))


def _get(url, timeout=20):
    req = urllib.request.Request(url, headers={"Authorization": os.environ["PEXELS_API_KEY"], "User-Agent": "MedReel/1.0"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _brief(v):
    return {"id": v["id"], "image": v.get("image"), "duration": v.get("duration"), "user": (v.get("user") or {}).get("name"),
            "url": v.get("url"), "width": v.get("width"), "height": v.get("height")}


def search(query, per_page=8, orientation="portrait"):
    q = urllib.parse.urlencode({"query": query[:80], "per_page": per_page, "orientation": orientation, "size": "medium"})
    vids = _get(f"{API}/search?{q}").get("videos", [])
    vids = [v for v in vids if 3 <= (v.get("duration") or 0) <= 60]
    return [_brief(v) | {"_files": v.get("video_files", [])} for v in vids]


def _pick_file(files):
    """Smallest file that is still >= 1280 px tall (portrait) or wide; else the largest."""
    mp4 = [f for f in files if (f.get("file_type") or "").endswith("mp4") and f.get("link")]
    good = [f for f in mp4 if max(f.get("width") or 0, f.get("height") or 0) >= 1280]
    if good: return min(good, key=lambda f: (f.get("width") or 0) * (f.get("height") or 0))
    return max(mp4, key=lambda f: (f.get("width") or 0) * (f.get("height") or 0)) if mp4 else None


def fetch(video, project_dir, max_s=8.0):
    """video: a search result (with _files) or a Pexels video id. -> clip id ('px<id>') or None."""
    if not isinstance(video, dict):
        v = _get(f"{API}/videos/{int(video)}"); video = _brief(v) | {"_files": v.get("video_files", [])}
    clip = f"px{int(video['id'])}"
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
    json.dump({"pexels_id": video["id"], "url": video.get("url"), "user": video.get("user")}, open(os.path.join(d, clip + ".json"), "w"))
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
            res = search(CO.stock_query(key), per_page=6)
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
