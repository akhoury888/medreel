"""Database-backed render queue.

The web process only enqueues. Workers (python worker.py — any number, on any machine that shares
the database and STORAGE_DIR) claim jobs atomically, run them and report progress. Paid plans get
priority. A job whose worker stops heart-beating for 10 minutes is re-queued."""
import datetime as dt, json, logging, os, socket, threading, time, traceback
from sqlalchemy import select, update, delete
from . import config as C
from .models import Session, Job, Project, User, Usage, now

log = logging.getLogger("medreel.jobs")


def enqueue(project, kind, payload=None, priority=0):
    Session.execute(update(Job).where(Job.project_id == project.id, Job.state == "queued", Job.kind.in_(("preview",)))
                    .values(state="canceled", message="Superseded"))  # one pending preview is enough
    j = Job(project_id=project.id, user_id=project.user_id, kind=kind, payload=json.dumps(payload or {}), priority=priority,
            message="Queued…")
    Session.add(j)
    project.status = "processing"
    Session.commit()
    return j


def latest(project_id):
    return Session.scalar(select(Job).where(Job.project_id == project_id).order_by(Job.id.desc()).limit(1))


def job_payload(j):
    if not j: return {"state": "idle"}
    return {"id": j.id, "state": j.state, "kind": j.kind, "progress": round(j.progress, 3), "msg": j.message,
            "result": json.loads(j.result or "{}")}


def _claim(worker_id):
    s = Session()
    stale = now() - dt.timedelta(minutes=10)
    s.execute(update(Job).where(Job.state == "working", Job.heartbeat < stale).values(state="queued", message="Re-queued after a worker stopped"))
    s.commit()
    for _ in range(5):
        j = s.scalar(select(Job).where(Job.state == "queued").order_by(Job.priority.desc(), Job.id).limit(1))
        if not j: return None
        r = s.execute(update(Job).where(Job.id == j.id, Job.state == "queued")
                      .values(state="working", worker=worker_id, started_at=now(), heartbeat=now(), message="Starting…"))
        s.commit()
        if r.rowcount == 1:
            s.refresh(j); return j
    return None


def _brand_for(user, plan_name):
    from engine import planner
    b = {**planner.DEFAULT_BRAND, "disclaimer": None}
    b.update({k: v for k, v in user.brand.items() if v not in (None, "")})
    if not b.get("name"): b["name"] = user.name or "Your name"
    if not b.get("specialty"): b["specialty"] = user.specialty or ""
    if b.get("disclaimer") is None: b.pop("disclaimer")
    return b


def run_job(j):
    from engine import pipeline
    s = Session()
    p = s.get(Project, j.project_id); u = s.get(User, j.user_id)
    if not p or not u:
        j.state = "canceled"; s.commit(); return
    last = [0.0]

    def prog(x, msg=None):
        t = time.time()
        if t - last[0] < 1.0 and x < 1: return
        last[0] = t
        s.execute(update(Job).where(Job.id == j.id).values(progress=float(min(1, max(0, x))), heartbeat=now(), **({"message": msg} if msg else {})))
        s.commit()
    payload = json.loads(j.payload or "{}")
    plan_cfg = C.PLANS[u.effective_plan()]
    try:
        if j.kind == "generate":
            src = payload.get("src")
            pipeline.create(p.dir, p.mode, src, p.script, p.style, _brand_for(u, plan_cfg), p.language, log=lambda m: None,
                            progress=lambda x, m: prog(x * 0.5, m))
            plan_path = os.path.join(p.dir, "plan.json")
            plan = json.load(open(plan_path))
            if p.title in ("", "Untitled video") and plan["title"].get("text"): p.title = plan["title"]["text"][:300]
            p.language = plan.get("language", p.language)
            s.commit()
            kind, base = "preview", 0.5
        else:
            kind, base = j.kind, 0.0
        plan_path = os.path.join(p.dir, "plan.json")
        plan = json.load(open(plan_path))
        plan["output"]["watermark"] = bool(plan_cfg["watermark"])
        json.dump(plan, open(plan_path, "w"), indent=1)
        prog(base, "Rendering the preview…" if kind == "preview" else "Rendering your final 1080×1920 video…")
        res = pipeline.render_project(p.dir, preview=(kind == "preview"), log=lambda m: None, progress=lambda x: prog(base + (1 - base) * x))
        p.duration = res["duration"]; p.status = "ready"
        if kind == "preview":  # dashboard thumbnail from the edited video (titles and graphics included)
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{min(4.0, res['duration'] / 2):.2f}", "-i", res["path"], "-frames:v", "1",
                            "-vf", "scale=360:-2", os.path.join(p.dir, "thumb.jpg")])
        if kind == "final": p.exported_at = now()
        msg = ("Preview ready." if kind == "preview" else "Your video is ready to download.")
        s.execute(update(Job).where(Job.id == j.id).values(state="done", progress=1.0, message=msg, finished_at=now(),
                                                           result=json.dumps({"duration": res["duration"], "seconds": res["seconds"], "kind": kind})))
        s.commit()
    except Exception as e:
        log.error("job %s failed\n%s", j.id, traceback.format_exc())
        s.rollback()
        p = s.get(Project, j.project_id)
        if j.kind == "generate":  # a failed generation gives the credit back
            s.execute(delete(Usage).where(Usage.project_id == j.project_id))
            if p: p.status = "error"
        elif p: p.status = "ready" if os.path.exists(os.path.join(p.dir, "plan.json")) else "error"
        friendly = str(e)[:300] if isinstance(e, (RuntimeError, ValueError)) else "Something went wrong while processing this video. Your credit was not used."
        s.execute(update(Job).where(Job.id == j.id).values(state="error", message=friendly, finished_at=now()))
        s.commit()
    finally:
        Session.remove()


def worker_loop(stop=None, idle=1.5):
    wid = f"{socket.gethostname()}:{os.getpid()}:{threading.get_ident() % 10000}"
    log.info("worker %s started", wid)
    while not (stop and stop.is_set()):
        try:
            j = _claim(wid)
        except Exception:
            log.exception("claim failed"); Session.remove(); time.sleep(5); continue
        if j: run_job(j)
        else:
            Session.remove(); time.sleep(idle)


def start_inline_worker():
    t = threading.Thread(target=worker_loop, daemon=True, name="inline-worker"); t.start(); return t
