"""All runtime configuration comes from environment variables (see .env.example)."""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def env(k, d=None): return os.environ.get(k, d)


SECRET_KEY = env("SECRET_KEY", "dev-insecure-change-me")
DATABASE_URL = env("DATABASE_URL", "sqlite:///" + os.path.join(BASE, "data", "medreel.db"))
STORAGE_DIR = env("STORAGE_DIR", os.path.join(BASE, "data", "storage"))
PUBLIC_URL = env("PUBLIC_URL", "http://localhost:8080").rstrip("/")
DEV_MODE = env("MEDREEL_DEV", "0") == "1"                # enables fake billing + shows reset links on screen
WORKER_INLINE = env("WORKER_INLINE", "1") == "1"         # run a render worker thread inside the web process
MAX_UPLOAD_MB = int(env("MAX_UPLOAD_MB", "1500"))
MAX_VIDEO_SECONDS = int(env("MAX_VIDEO_SECONDS", "600"))  # 10-minute recordings

# Stripe
STRIPE_SECRET_KEY = env("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = env("STRIPE_WEBHOOK_SECRET")
STRIPE_AUTOMATIC_TAX = env("STRIPE_AUTOMATIC_TAX", "0") == "1"

# Email (password reset). If unset, reset links are logged (and shown on screen in DEV_MODE).
SMTP_HOST, SMTP_PORT = env("SMTP_HOST"), int(env("SMTP_PORT", "587"))
SMTP_USER, SMTP_PASSWORD, MAIL_FROM = env("SMTP_USER"), env("SMTP_PASSWORD"), env("MAIL_FROM", "MedReel <no-reply@localhost>")

# Plans. Prices are shown on the site; what is actually charged is the Stripe Price object
# (STRIPE_PRICE_<PLAN>). Change both together.
PLANS = {
    "free":    {"name": "Free",    "price": 0,   "videos": 1,   "watermark": True,  "priority": False, "seats": 1,
                "features": ["1 video per month", "All 4 creation modes", "Arabic, English, French", "MedReel watermark"]},
    "starter": {"name": "Starter", "price": 49,  "videos": 8,   "watermark": False, "priority": False, "seats": 1,
                "features": ["8 videos per month", "No watermark", "Your branding and logo", "Subtitles in 3 languages"]},
    "pro":     {"name": "Pro",     "price": 129, "videos": 30,  "watermark": False, "priority": True,  "seats": 1,
                "features": ["30 videos per month", "Priority rendering", "AI scripts and teleprompter", "Everything in Starter"]},
    "clinic":  {"name": "Clinic",  "price": 349, "videos": 100, "watermark": False, "priority": True,  "seats": 5,
                "features": ["100 videos per month", "Up to 5 physicians (coming soon)", "Priority rendering", "Everything in Pro"]},
}
CURRENCY = env("PRICE_CURRENCY", "USD")
for _k in PLANS:
    PLANS[_k]["stripe_price"] = env(f"STRIPE_PRICE_{_k.upper()}")
ACTIVE_STATUSES = {"active", "trialing", "past_due"}  # past_due keeps access while Stripe retries the card
