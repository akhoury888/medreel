"""Project API (every route is scoped to the signed-in user)."""
import json, os, re, shutil, subprocess
from flask import Blueprint, request, jsonify, abort, send_from_directory
from sqlalchemy import select
from . import config as C, jobs
from .auth import current_user, login_required
from .models import Session, Project, Usage, Job, videos_used, now

bp = Blueprint("api", __name__)
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".3gp"}


def _own(pid):
    u = current_user(); p = Session.get(Project, pid)
    if not p or p.user_id != u.id: abort(404)
    return p


def _plan_path(p):
    path = os.path.join(p.dir, "plan.json")
    if not os.path.exists(path): abort(409, "This video is still being prepared.")
    return path


def _probe(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                             capture_output=True, timeout=60).stdout
        j = json.loads(out or b"{}")
    except Exception:
        return None
    types = {s.get("codec_type") for s in j.get("streams", [])}
    dur = float(j.get("format", {}).get("duration") or 0)
    if not dur:  # MediaRecorder webm often has no duration header
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "packet=pts_time", "-select_streams", "a:0", "-of", "csv=p=0", path],
                               capture_output=True, timeout=120, text=True).stdout.split()
            dur = float(r[-1]) if r else 0
        except Exception: pass
    return {"video": "video" in types, "audio": "audio" in types, "duration": dur}


def _stock_credit():
    from engine import broll
    c = broll.credit()
    return {"name": c[0], "url": c[1]} if c else None


@bp.get("/api/meta")
@login_required
def meta():
    from engine import planner, graphics, pipeline, synth
    return jsonify({"styles": {k: v["label"] for k, v in planner.STYLES.items()}, "layouts": planner.LAYOUTS,
                    "captions": planner.CAPTION_STYLES, "music": planner.MUSIC_TRACKS, "assets": graphics.ASSET_INFO,
                    "brand_defaults": planner.DEFAULT_BRAND, "modes": pipeline.MODES,
                    "voices": {l: synth.available(l) for l in ("en", "fr", "ar")},
                    "ai_scripts": bool(os.environ.get("ANTHROPIC_API_KEY")), "stock": _stock_credit(), "ai_images": bool(os.environ.get("FAL_KEY")),
                    "projects": [p.id for p in Session.scalars(select(Project).where(Project.user_id == current_user().id).order_by(Project.created_at.desc()).limit(20))]})


@bp.get("/api/projects")
@login_required
def list_projects():
    u = current_user()
    ps = Session.scalars(select(Project).where(Project.user_id == u.id).order_by(Project.created_at.desc())).all()
    out = []
    for p in ps:
        j = jobs.latest(p.id)
        out.append({"id": p.id, "title": p.title, "mode": p.mode, "language": p.language, "status": p.status, "duration": p.duration,
                    "created_at": p.created_at.isoformat(), "exported": bool(p.exported_at),
                    "has_thumb": os.path.exists(os.path.join(p.dir, "thumb.jpg")), "job": jobs.job_payload(j)})
    return jsonify({"projects": out})


@bp.post("/api/projects")
@login_required
def create_project():
    from engine import pipeline, planner
    u = current_user(); f = request.form
    mode = f.get("mode", "auto_edit")
    if mode not in pipeline.MODES: abort(400, "Unknown creation mode.")
    plan_cfg = C.PLANS[u.effective_plan()]
    used = videos_used(Session, u)
    if used >= plan_cfg["videos"]:
        return jsonify({"error": f"You've used all {plan_cfg['videos']} video(s) included in the {plan_cfg['name']} plan this month. Upgrade to make more.",
                        "upgrade": True}), 402
    script = (f.get("script") or "").strip()[:12000]
    language = f.get("language", u.default_language)
    if language not in ("auto", "en", "fr", "ar"): language = "auto"
    style = f.get("style", u.default_style)
    if style not in planner.STYLES: style = "modern_educational"
    video = request.files.get("video")
    if mode in ("auto_edit", "own_script", "teleprompter") and not video: abort(400, "Add your recording.")
    if mode in ("own_script", "teleprompter", "ai_voice") and not script: abort(400, "Add the script.")
    if mode == "ai_voice" and len(script.split()) > 450: abort(400, "That script is too long for one video (keep it under ~3 minutes).")
    p = Project(user_id=u.id, title=(f.get("title") or "").strip()[:300] or "Untitled video", mode=mode, language=language,
                style=style, script=script)
    Session.add(p); Session.flush()
    os.makedirs(os.path.join(p.dir, "renders"), exist_ok=True)
    payload = {}
    if video:
        ext = os.path.splitext(video.filename or "")[1].lower() or ".mp4"
        if ext not in VIDEO_EXT: Session.rollback(); shutil.rmtree(p.dir, ignore_errors=True); abort(400, "Upload a video file (MP4, MOV or WebM).")
        up = os.path.join(p.dir, "upload" + ext); video.save(up)
        info = _probe(up)
        err = None
        if not info or not info["audio"]: err = "We couldn't find any sound in this file. Record with your microphone on."
        elif not info["video"]: err = "This file has no video track."
        elif info["duration"] > C.MAX_VIDEO_SECONDS + 1: err = f"Recordings can be up to {C.MAX_VIDEO_SECONDS // 60} minutes."
        if err: Session.rollback(); shutil.rmtree(p.dir, ignore_errors=True); abort(400, err)
        if ext == ".webm":  # browser recordings: normalise to MP4 (constant frame rate) once
            mp4 = os.path.join(p.dir, "upload.mp4")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", up, "-vf", "fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                            "-c:a", "aac", "-b:a", "192k", mp4], check=True)
            os.remove(up); up = mp4
        _, _, src = pipeline.ingest(os.path.dirname(p.dir), up, p.id); os.remove(up)
        payload["src"] = src
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", src, "-frames:v", "1", "-vf", "scale=360:-2", os.path.join(p.dir, "thumb.jpg")])
    Session.add(Usage(user_id=u.id, project_id=p.id)); p.counted = True
    Session.commit()
    jobs.enqueue(p, "generate", payload, priority=1 if plan_cfg["priority"] else 0)
    return jsonify({"id": p.id, "usage": {"used": used + 1, "limit": plan_cfg["videos"]}})


@bp.get("/api/projects/<pid>")
@login_required
def get_project(pid):
    from engine import planner
    p = _own(pid)
    path = os.path.join(p.dir, "plan.json")
    plan = json.load(open(path)) if os.path.exists(path) else None
    if plan:
        tl = planner.Timeline(planner.keep_segments(plan))
        plan["_out"] = {"duration": tl.duration, "removed": round(plan["source"]["duration"] - tl.duration, 2),
                        "cuts": {c["id"]: tl.to_out(c["start"]) for c in plan["cuts"]},
                        "scenes": {s["id"]: [tl.to_out(s["start"]), tl.to_out(s["end"])] for s in plan["scenes"]}}
        plan["source"]["path"] = "source" + os.path.splitext(plan["source"]["path"])[1]  # never leak server paths
        plan["branding"].pop("logo", None)
    hist = path + ".history.json"
    return jsonify({"id": p.id, "title": p.title, "mode": p.mode, "status": jobs.job_payload(jobs.latest(p.id)), "plan": plan,
                    "renders": {k: os.path.exists(os.path.join(p.dir, "renders", k + ".mp4")) for k in ("preview", "final")},
                    "can_undo": os.path.exists(hist) and len(json.load(open(hist))) > 0})


@bp.get("/api/projects/<pid>/status")
@login_required
def status(pid):
    return jsonify(jobs.job_payload(jobs.latest(_own(pid).id)))


@bp.patch("/api/projects/<pid>")
@login_required
def rename(pid):
    p = _own(pid); t = ((request.get_json(force=True, silent=True) or {}).get("title") or "").strip()
    if t: p.title = t[:300]; Session.commit()
    return jsonify({"ok": True, "title": p.title})


def _busy(p):
    j = jobs.latest(p.id)
    return j and j.state in ("queued", "working") and j.kind == "generate"


def _prio(): return 1 if C.PLANS[current_user().effective_plan()]["priority"] else 0


@bp.post("/api/projects/<pid>/command")
@login_required
def command(pid):
    from engine import commands
    p = _own(pid)
    if _busy(p): abort(409, "Wait until the first edit is finished.")
    body = request.get_json(force=True, silent=True) or {}
    res = commands.execute(_plan_path(p), (body.get("text") or "")[:500])
    if res["ok"] and body.get("render", True): jobs.enqueue(p, "preview", priority=_prio())
    return jsonify(res)


@bp.post("/api/projects/<pid>/ops")
@login_required
def ops(pid):
    from engine import commands
    p = _own(pid)
    if _busy(p): abort(409, "Wait until the first edit is finished.")
    body = request.get_json(force=True, silent=True) or {}
    for o in body.get("ops", []):  # brand changes go through the account profile, not file paths
        if o.get("op") == "set_branding": o["fields"] = {k: v for k, v in (o.get("fields") or {}).items() if k != "logo"}
    res = commands.execute_ops(_plan_path(p), body.get("ops", []))
    if body.get("render"): jobs.enqueue(p, "preview", priority=_prio())
    return jsonify(res)


@bp.post("/api/projects/<pid>/photos")
@login_required
def add_photo(pid):
    """Upload a photo (JPG/PNG/WebP) and place it in the video at a phrase or at the current preview time."""
    import uuid
    from PIL import Image, ImageOps
    from engine import commands
    p = _own(pid)
    if _busy(p): abort(409, "Wait until the first edit is finished.")
    f = request.files.get("photo")
    if not f: abort(400, "Choose a photo.")
    data = f.read(15 * 1024 * 1024 + 1)
    if len(data) > 15 * 1024 * 1024: abort(400, "Photos can be up to 15 MB.")
    try:
        import io
        im = Image.open(io.BytesIO(data)); im.verify()
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        abort(400, "That file isn't a photo we can read. Use JPG, PNG or WebP.")
    im.thumbnail((2000, 2000))
    os.makedirs(os.path.join(p.dir, "photos"), exist_ok=True)
    photo_id = uuid.uuid4().hex[:12]
    im.save(os.path.join(p.dir, "photos", photo_id + ".jpg"), "JPEG", quality=90)  # re-encoded: no EXIF/location data kept
    if request.form.get("place") == "0":  # just add it to this video's photo library
        return jsonify({"ok": True, "photo": photo_id, "message": "Photo uploaded — now choose where it goes."})
    op = {"op": "add_photo", "photo": photo_id, "caption": (request.form.get("caption") or "").strip()[:80],
          "phrase": (request.form.get("phrase") or "").strip()[:200]}
    if (request.form.get("sentence") or "").isdigit(): op["sentence"] = int(request.form["sentence"])
    try: op["t_out"] = max(0.0, float(request.form.get("t_out") or 0))
    except ValueError: pass
    res = commands.execute_ops(_plan_path(p), [op], label="Add photo")
    jobs.enqueue(p, "preview", priority=_prio())
    return jsonify({**res, "photo": photo_id})


@bp.get("/api/projects/<pid>/photos")
@login_required
def list_photos(pid):
    p = _own(pid); d = os.path.join(p.dir, "photos")
    ids = sorted((f[:-4] for f in os.listdir(d) if f.endswith(".jpg")), key=lambda i: os.path.getmtime(os.path.join(d, i + ".jpg"))) if os.path.isdir(d) else []
    return jsonify({"photos": ids})


@bp.get("/api/stock/search")
@login_required
def stock_search():
    from engine import broll
    if not broll.available(): return jsonify({"error": "Stock videos need a free Pixabay or Pexels key.", "results": []}), 503
    q = (request.args.get("q") or "").strip()[:80]
    if not q: return jsonify({"results": []})
    try: res = broll.search(q, per_page=12)
    except Exception: return jsonify({"error": "The stock video library didn't answer. Try again in a moment.", "results": []}), 502
    return jsonify({"results": [{k: v for k, v in r.items() if k != "_files"} for r in res]})


@bp.post("/api/projects/<pid>/stock")
@login_required
def add_stock(pid):
    from engine import broll, commands
    p = _own(pid)
    if _busy(p): abort(409, "Wait until the first edit is finished.")
    b = request.get_json(force=True, silent=True) or {}
    vid = str(b.get("video_id") or "")
    if not broll.CLIP_RE.fullmatch(vid): abort(400, "Choose a video.")
    try: clip = broll.fetch(vid, p.dir)
    except Exception: clip = None
    if not clip: abort(502, "Couldn't download that video. Try another one.")
    op = {"op": "add_broll", "clip": clip}
    if str(b.get("sentence", "")).isdigit(): op["sentence"] = int(b["sentence"])
    else:
        try: op["t_out"] = max(0.0, float(b.get("t_out") or 0))
        except (TypeError, ValueError): pass
    res = commands.execute_ops(_plan_path(p), [op], label="Add stock video")
    jobs.enqueue(p, "preview", priority=_prio())
    return jsonify(res)


@bp.post("/api/projects/<pid>/ai_image")
@login_required
def ai_image(pid):
    """Generate an illustration for one sentence (Claude writes the brief unless the doctor typed one) and place it there.
    replace=<scene id> swaps an existing AI image for a new one."""
    from engine import imagegen, commands
    p = _own(pid)
    if _busy(p): abort(409, "Wait until the first edit is finished.")
    if not imagegen.available(): abort(503, "AI illustrations need a fal.ai key (FAL_KEY).")
    b = request.get_json(force=True, silent=True) or {}
    plan = json.load(open(_plan_path(p)))
    sents = plan["analysis"]["sentences"]
    try: i = int(b.get("sentence"))
    except (TypeError, ValueError): abort(400, "Choose a sentence.")
    if not 0 <= i < len(sents): abort(400, "Choose a sentence.")
    brief = (b.get("brief") or "").strip()[:500]
    if not brief:
        try: brief = imagegen.brief_for_sentence(sents[i]["text"], plan["branding"].get("specialty", ""))
        except Exception: brief = None
    if not brief:  # no Claude key: use the concept vocabulary
        from engine import concepts as CO
        hits = CO.find(plan["transcript"][sents[i]["i0"]: sents[i]["i1"] + 1])
        if not hits: abort(400, "Describe the image you want in the box above (or add a Claude key so it can write the description for you).")
        brief = imagegen.concept_brief(hits[0][1])
    try:
        photo = imagegen.generate(brief, p.dir, plan["branding"].get("image_style", "3d"), plan["branding"].get("primary", "#2EA0FF"))
    except Exception as e:
        abort(502, f"The image service didn't answer ({str(e)[:80]}). Try again.")
    ops = []
    if b.get("replace"): ops.append({"op": "delete_scene", "id": str(b["replace"])[:40]})
    ops.append({"op": "add_photo", "photo": photo, "sentence": i, "fill": True, "caption": ""})
    res = commands.execute_ops(_plan_path(p), ops, label="AI illustration")
    jobs.enqueue(p, "preview", priority=_prio())
    return jsonify({**res, "photo": photo, "brief": brief})


@bp.post("/api/projects/<pid>/undo")
@login_required
def undo(pid):
    from engine import planner
    p = _own(pid); prev = planner.undo(_plan_path(p))
    if prev: jobs.enqueue(p, "preview", priority=_prio())
    return jsonify({"ok": bool(prev)})


@bp.post("/api/projects/<pid>/render")
@login_required
def render(pid):
    p = _own(pid); _plan_path(p)
    final = bool((request.get_json(force=True, silent=True) or {}).get("final"))
    jobs.enqueue(p, "final" if final else "preview", priority=_prio())
    return jsonify({"ok": True})


@bp.delete("/api/projects/<pid>")
@login_required
def delete(pid):
    from engine import pipeline
    p = _own(pid)
    if os.path.isdir(p.dir): pipeline.delete_project(p.dir)
    Session.query(Job).filter(Job.project_id == p.id, Job.state.in_(("queued",))).update({"state": "canceled"})
    Session.delete(p); Session.commit()
    return jsonify({"ok": True, "message": "Recording, renders, transcript and plan permanently deleted."})


@bp.get("/media/<pid>/<name>")
@login_required
def media(pid, name):
    p = _own(pid)
    allowed = {"preview.mp4": "renders", "final.mp4": "renders", "plan.json": ".", "thumb.jpg": "."}
    allowed.update({f"{k}{x}.srt": "renders" for k in ("preview", "final") for x in ("", ".en", ".fr", ".ar")})
    if name.startswith("source.") and "/" not in name: return send_from_directory(p.dir, name, conditional=True)
    m = re.fullmatch(r"photo_([0-9a-f]{12})\.jpg", name)
    if m: return send_from_directory(os.path.join(p.dir, "photos"), m.group(1) + ".jpg", conditional=True)
    m = re.fullmatch(r"broll_(p[xb]\d{1,12})\.jpg", name)
    if m: return send_from_directory(os.path.join(p.dir, "broll"), m.group(1) + ".jpg", conditional=True)
    if name not in allowed: abort(404)
    dl = request.args.get("dl") == "1"
    nice = f"{(p.title or 'video')[:60]}{os.path.splitext(name)[1]}" if dl else None
    return send_from_directory(os.path.join(p.dir, allowed[name]), name, conditional=True, as_attachment=dl, download_name=nice)


@bp.post("/api/scripts/generate")
@login_required
def gen_script():
    from engine import scripts
    b = request.get_json(force=True, silent=True) or {}; u = current_user()
    lang = b.get("language") if b.get("language") in ("en", "fr", "ar") else "en"
    try:
        out = scripts.generate((b.get("prompt") or "")[:1000], lang, int(b.get("seconds") or 45), u.specialty or "cardiology", u.name)
    except scripts.ScriptUnavailable as e:
        return jsonify({"error": str(e), "templates": scripts.templates(lang)}), 503
    except Exception:
        return jsonify({"error": "The script writer is unavailable right now. Try again, or start from a template.", "templates": scripts.templates(lang)}), 502
    return jsonify(out)


@bp.get("/api/scripts/templates")
@login_required
def templates():
    from engine import scripts
    lang = request.args.get("language") if request.args.get("language") in ("en", "fr", "ar") else "en"
    return jsonify({"templates": scripts.templates(lang)})
