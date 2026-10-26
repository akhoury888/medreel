"""Accounts: signup, login, logout, password reset/change, profile & branding, account deletion."""
import functools, json, logging, os, re, secrets, shutil, smtplib, time
from email.message import EmailMessage
from flask import Blueprint, request, session, jsonify, g, abort
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import select
from . import config as C
from .models import Session, User, videos_used

bp = Blueprint("auth", __name__)
log = logging.getLogger("medreel.auth")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)
_attempts = {}  # (ip|email) -> [timestamps]   simple brute-force throttle (per process)


def _throttle(key, limit=8, window=300):
    t = time.time(); xs = [x for x in _attempts.get(key, []) if t - x < window]
    _attempts[key] = xs
    if len(xs) >= limit: abort(429, "Too many attempts. Wait a few minutes and try again.")
    xs.append(t)


def _serializer(): return URLSafeTimedSerializer(C.SECRET_KEY, salt="pw-reset")


def current_user():
    if "user" in g: return g.user
    uid, sv = session.get("uid"), session.get("sv")
    u = Session.get(User, uid) if uid else None
    g.user = u if (u and u.session_version == sv) else None
    return g.user


def login_required(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        if not current_user(): return jsonify({"error": "Please sign in."}), 401
        return fn(*a, **k)
    return w


def _login(u):
    session.clear(); session.permanent = True
    session.update(uid=u.id, sv=u.session_version, csrf=secrets.token_urlsafe(24))


def user_payload(u):
    plan = u.effective_plan(); p = C.PLANS[plan]
    return {"id": u.id, "email": u.email, "name": u.name, "specialty": u.specialty, "brand": u.brand,
            "default_language": u.default_language, "default_style": u.default_style,
            "plan": plan, "plan_name": p["name"], "subscribed_plan": u.plan, "sub_status": u.sub_status,
            "current_period_end": u.current_period_end.isoformat() if u.current_period_end else None,
            "cancel_at_period_end": u.cancel_at_period_end,
            "usage": {"used": videos_used(Session, u), "limit": p["videos"], "watermark": p["watermark"]},
            "csrf": session.get("csrf"), "billing_live": bool(C.STRIPE_SECRET_KEY), "dev": C.DEV_MODE}


def _validate_pw(pw):
    if len(pw or "") < 10: abort(400, "Use a password of at least 10 characters.")


@bp.post("/api/auth/signup")
def signup():
    b = request.get_json(force=True, silent=True) or {}
    email = (b.get("email") or "").strip().lower()
    _throttle("signup:" + (request.remote_addr or ""), limit=10, window=3600)
    if not EMAIL_RE.match(email): abort(400, "Enter a valid email address.")
    _validate_pw(b.get("password"))
    if Session.scalar(select(User).where(User.email == email)): abort(409, "An account with this email already exists. Sign in instead.")
    name = (b.get("name") or "").strip()[:200]
    u = User(email=email, pw_hash=generate_password_hash(b["password"]), name=name,
             specialty=(b.get("specialty") or "Cardiology").strip()[:200],
             brand_json=json.dumps({"name": name, "specialty": (b.get("specialty") or "Cardiology").strip()[:200]}))
    Session.add(u); Session.commit(); _login(u)
    return jsonify(user_payload(u))


@bp.post("/api/auth/login")
def login():
    b = request.get_json(force=True, silent=True) or {}
    email = (b.get("email") or "").strip().lower()
    _throttle("login:" + email); _throttle("loginip:" + (request.remote_addr or ""), limit=30)
    u = Session.scalar(select(User).where(User.email == email))
    if not u or not check_password_hash(u.pw_hash, b.get("password") or ""): abort(401, "Wrong email or password.")
    _login(u)
    return jsonify(user_payload(u))


@bp.post("/api/auth/logout")
def logout():
    session.clear(); return jsonify({"ok": True})


@bp.get("/api/auth/me")
def me():
    u = current_user()
    if not u: return jsonify({"user": None, "plans": public_plans(), "dev": C.DEV_MODE})
    return jsonify({"user": user_payload(u), "plans": public_plans()})


def public_plans():
    return {k: {x: v[x] for x in ("name", "price", "videos", "watermark", "features")} | {"available": k == "free" or bool(v["stripe_price"]) or C.DEV_MODE}
            for k, v in C.PLANS.items()} | {"_currency": C.CURRENCY}


def send_mail(to, subject, body):
    if not C.SMTP_HOST:
        log.warning("SMTP not configured — email to %s: %s\n%s", to, subject, body); return False
    m = EmailMessage(); m["From"], m["To"], m["Subject"] = C.MAIL_FROM, to, subject; m.set_content(body)
    with smtplib.SMTP(C.SMTP_HOST, C.SMTP_PORT, timeout=20) as s:
        s.starttls()
        if C.SMTP_USER: s.login(C.SMTP_USER, C.SMTP_PASSWORD)
        s.send_message(m)
    return True


@bp.post("/api/auth/forgot")
def forgot():
    b = request.get_json(force=True, silent=True) or {}
    email = (b.get("email") or "").strip().lower()
    _throttle("forgot:" + (request.remote_addr or ""), limit=5, window=900)
    u = Session.scalar(select(User).where(User.email == email))
    out = {"ok": True, "message": "If an account exists for this email, a reset link is on its way."}
    if u:
        tok = _serializer().dumps({"uid": u.id, "h": u.pw_hash[-12:]})
        link = f"{C.PUBLIC_URL}/app#/reset/{tok}"
        sent = send_mail(u.email, "Reset your MedReel password", f"Reset your password (valid for 1 hour):\n{link}\n\nIf you didn't ask for this, ignore this email.")
        if C.DEV_MODE and not sent: out["dev_link"] = link
    return jsonify(out)


@bp.post("/api/auth/reset")
def reset():
    b = request.get_json(force=True, silent=True) or {}
    try: data = _serializer().loads(b.get("token") or "", max_age=3600)
    except SignatureExpired: abort(400, "This reset link has expired. Ask for a new one.")
    except BadSignature: abort(400, "This reset link is not valid.")
    u = Session.get(User, data["uid"])
    if not u or u.pw_hash[-12:] != data["h"]: abort(400, "This reset link was already used.")
    _validate_pw(b.get("password"))
    u.pw_hash = generate_password_hash(b["password"]); u.session_version += 1; Session.commit(); _login(u)
    return jsonify(user_payload(u))


@bp.post("/api/auth/password")
@login_required
def change_password():
    b = request.get_json(force=True, silent=True) or {}; u = current_user()
    if not check_password_hash(u.pw_hash, b.get("current") or ""): abort(400, "Current password is wrong.")
    _validate_pw(b.get("new"))
    u.pw_hash = generate_password_hash(b["new"]); u.session_version += 1; Session.commit(); _login(u)
    return jsonify({"ok": True, "message": "Password changed. Other devices were signed out."})


BRAND_KEYS = ("name", "specialty", "clinic", "website", "booking", "phone", "primary", "secondary", "disclaimer", "lower_third", "outro", "image_style")


@bp.put("/api/account/profile")
@login_required
def profile():
    from engine import planner
    b = request.get_json(force=True, silent=True) or {}; u = current_user()
    if "name" in b: u.name = str(b["name"])[:200]
    if "specialty" in b: u.specialty = str(b["specialty"])[:200]
    if b.get("default_language") in ("auto", "en", "fr", "ar"): u.default_language = b["default_language"]
    if b.get("default_style") in planner.STYLES: u.default_style = b["default_style"]
    if isinstance(b.get("brand"), dict):
        br = u.brand
        for k in BRAND_KEYS:
            if k in b["brand"]:
                v = b["brand"][k]
                if k in ("primary", "secondary") and not re.fullmatch(r"#[0-9a-fA-F]{6}", str(v)): continue
                if k == "image_style" and v not in ("3d", "flat", "photo"): continue
                br[k] = v if isinstance(v, bool) else str(v)[:300]
        u.brand_json = json.dumps(br)
    Session.commit()
    return jsonify(user_payload(u))


@bp.post("/api/account/logo")
@login_required
def logo():
    from PIL import Image
    u = current_user(); f = request.files.get("logo")
    if not f: abort(400, "No file.")
    d = os.path.join(C.STORAGE_DIR, "users", u.id); os.makedirs(d, exist_ok=True)
    try:
        im = Image.open(f.stream).convert("RGBA"); im.thumbnail((600, 600))
    except Exception:
        abort(400, "Upload a PNG or JPG image.")
    p = os.path.join(d, "logo.png"); im.save(p)
    br = u.brand; br["logo"] = p; u.brand_json = json.dumps(br); Session.commit()
    return jsonify(user_payload(u))


@bp.delete("/api/account/logo")
@login_required
def logo_delete():
    u = current_user(); br = u.brand; p = br.pop("logo", None)
    if p and os.path.exists(p): os.remove(p)
    u.brand_json = json.dumps(br); Session.commit(); return jsonify(user_payload(u))


@bp.delete("/api/account")
@login_required
def delete_account():
    """Permanent deletion: cancels the subscription, deletes every recording, render, transcript and the account."""
    from .billing import cancel_now
    b = request.get_json(force=True, silent=True) or {}; u = current_user()
    if not check_password_hash(u.pw_hash, b.get("password") or ""): abort(400, "Password is wrong.")
    cancel_now(u)
    d = os.path.join(C.STORAGE_DIR, "users", u.id)
    if os.path.isdir(d):
        for root, _, files in os.walk(d):
            for f in files: os.chmod(os.path.join(root, f), 0o600)
        shutil.rmtree(d)
    Session.delete(u); Session.commit(); session.clear()
    return jsonify({"ok": True, "message": "Your account and all your videos were permanently deleted."})
