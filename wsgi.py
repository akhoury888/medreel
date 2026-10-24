"""MedReel web app.   Local:  MEDREEL_DEV=1 python wsgi.py   ->  http://localhost:8080
Production: gunicorn -w 2 -k gthread --threads 8 -b 0.0.0.0:$PORT wsgi:app   (+ python worker.py)"""
import datetime as dt, logging, os
from flask import Flask, jsonify, request, session, send_from_directory, redirect
from werkzeug.exceptions import HTTPException
from saas import config as C
from saas.models import init_db, Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
CSRF_EXEMPT = {"/stripe/webhook", "/api/auth/login", "/api/auth/signup", "/api/auth/forgot", "/api/auth/reset"}


def create_app():
    if C.SECRET_KEY == "dev-insecure-change-me" and not C.DEV_MODE:
        raise SystemExit("Set SECRET_KEY (or MEDREEL_DEV=1 for local testing).")
    os.makedirs(C.STORAGE_DIR, exist_ok=True)
    init_db()
    app = Flask(__name__, static_folder="web", static_url_path="/static")
    app.config.update(SECRET_KEY=C.SECRET_KEY, MAX_CONTENT_LENGTH=C.MAX_UPLOAD_MB * 1024 * 1024,
                      SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                      SESSION_COOKIE_SECURE=C.PUBLIC_URL.startswith("https"), PERMANENT_SESSION_LIFETIME=dt.timedelta(days=30))
    from saas import auth, billing, api
    app.register_blueprint(auth.bp); app.register_blueprint(billing.bp); app.register_blueprint(api.bp)

    @app.before_request
    def csrf():
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and request.path not in CSRF_EXEMPT:
            tok = session.get("csrf")
            if not tok or request.headers.get("X-CSRF-Token") != tok:
                return jsonify({"error": "Your session expired — reload the page."}), 403

    @app.after_request
    def headers(r):
        r.headers.setdefault("X-Content-Type-Options", "nosniff")
        r.headers.setdefault("X-Frame-Options", "DENY")
        r.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        r.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(self)")
        if C.PUBLIC_URL.startswith("https"): r.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
        return r

    @app.errorhandler(HTTPException)
    def http_err(e):
        if request.path.startswith(("/api/", "/media/", "/stripe/")):
            return jsonify({"error": e.description if e.code != 413 else f"File too large (max {C.MAX_UPLOAD_MB} MB)."}), e.code
        return e

    @app.teardown_appcontext
    def cleanup(_): Session.remove()

    @app.get("/")
    @app.get("/app")
    def spa(): return send_from_directory("web", "app.html")

    @app.get("/studio")
    def studio(): return send_from_directory("web", "studio.html")

    @app.get("/healthz")
    def health(): return {"ok": True}

    if C.WORKER_INLINE:
        from saas.jobs import start_inline_worker
        start_inline_worker()
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), threaded=True)
