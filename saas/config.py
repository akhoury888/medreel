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
    "free":    {"name": "Free",    "price": 0,   "videos": 1,   "watermark": True,  "priority": False, "seats": 1, "ai_credits": 10,
                "features": ["1 video per month", "10 AI credits (illustrations)", "Video, in-browser or voice-only", "Arabic, English, French", "MedReel watermark"]},
    "starter": {"name": "Starter", "price": 49,  "videos": 8,   "watermark": False, "priority": False, "seats": 1, "ai_credits": 150,
                "features": ["8 videos per month", "150 AI credits for illustrations & animations", "No watermark", "Your branding and logo", "Subtitles in 3 languages"]},
    "pro":     {"name": "Pro",     "price": 129, "videos": 30,  "watermark": False, "priority": True,  "seats": 1, "ai_credits": 500,
                "features": ["30 videos per month", "500 AI credits for illustrations & animations", "Priority rendering", "Everything in Starter"]},
    "clinic":  {"name": "Clinic",  "price": 349, "videos": 100, "watermark": False, "priority": True,  "seats": 5, "ai_credits": 1500,
                "features": ["100 videos per month", "1,500 AI credits for illustrations & animations", "Up to 5 physicians (coming soon)", "Priority rendering", "Everything in Pro"]},
}
CURRENCY = env("PRICE_CURRENCY", "USD")
for _k in PLANS:  # AI credit allowance can be tuned without code changes: AI_CREDITS_STARTER=200 …
    PLANS[_k]["ai_credits"] = int(env(f"AI_CREDITS_{_k.upper()}", PLANS[_k]["ai_credits"]))
# What each AI action costs in credits (1 credit ≈ US$0.03 of fal.ai cost) and the hard ceiling per video
AI_COST = {"image": int(env("AI_COST_IMAGE", "1")), "animation": int(env("AI_COST_ANIMATION", "10"))}
AI_PER_VIDEO = {"image": int(env("AI_MAX_IMAGES_PER_VIDEO", "12")), "animation": int(env("AI_MAX_ANIMATIONS_PER_VIDEO", "4"))}
for _k in PLANS:
    PLANS[_k]["stripe_price"] = env(f"STRIPE_PRICE_{_k.upper()}")
ACTIVE_STATUSES = {"active", "trialing", "past_due"}  # past_due keeps access while Stripe retries the card
