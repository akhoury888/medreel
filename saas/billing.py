"""Stripe subscriptions: Checkout, Customer Portal, webhooks (idempotent, signature-verified).

Source of truth for a user's plan = Stripe subscription state delivered by webhooks.
MEDREEL_DEV=1 adds a fake checkout so the whole app can be tried without Stripe keys."""
import datetime as dt, logging
from flask import Blueprint, request, jsonify, abort
from sqlalchemy import select
from . import config as C
from .auth import current_user, login_required, user_payload
from .models import Session, User, StripeEvent, now

bp = Blueprint("billing", __name__)
log = logging.getLogger("medreel.billing")


def _stripe():
    import stripe
    if not C.STRIPE_SECRET_KEY: abort(503, "Payments are not configured yet.")
    stripe.api_key = C.STRIPE_SECRET_KEY
    return stripe


def _plan_for_price(price_id):
    for k, v in C.PLANS.items():
        if v.get("stripe_price") and v["stripe_price"] == price_id: return k
    return None


def _ts(x): return dt.datetime.fromtimestamp(int(x), dt.timezone.utc).replace(tzinfo=None) if x else None


def sync_subscription(user, sub):
    """Copy a Stripe Subscription object onto the user."""
    items = (sub.get("items") or {}).get("data") or []
    price = items[0]["price"]["id"] if items else None
    plan = _plan_for_price(price)
    status = sub.get("status") or ""
    user.stripe_subscription_id = sub.get("id") or ""
    user.sub_status = status
    user.cancel_at_period_end = bool(sub.get("cancel_at_period_end"))
    user.current_period_end = _ts(sub.get("current_period_end") or (items[0].get("current_period_end") if items else None))
    if status in ("canceled", "incomplete_expired") or not plan:
        user.plan = "free" if status in ("canceled", "incomplete_expired") else user.plan
    else:
        user.plan = plan
    if not plan and status not in ("canceled", "incomplete_expired"):
        log.error("subscription %s has unknown price %s — set STRIPE_PRICE_<PLAN>", sub.get("id"), price)


@bp.post("/api/billing/checkout")
@login_required
def checkout():
    plan = (request.get_json(force=True, silent=True) or {}).get("plan")
    if plan not in C.PLANS or plan == "free": abort(400, "Unknown plan.")
    u = current_user()
    if u.effective_plan() != "free" and u.stripe_subscription_id:
        return jsonify({"url": None, "portal": True, "message": "You already have a subscription — change plans in the billing portal."})
    if not C.STRIPE_SECRET_KEY:
        if C.DEV_MODE: return jsonify({"dev": True})
        abort(503, "Payments are not configured yet.")
    price = C.PLANS[plan]["stripe_price"]
    if not price: abort(503, f"Plan '{plan}' has no Stripe price configured.")
    stripe = _stripe()
    if not u.stripe_customer_id:
        c = stripe.Customer.create(email=u.email, name=u.name or None, metadata={"user_id": u.id})
        u.stripe_customer_id = c["id"]; Session.commit()
    params = dict(mode="subscription", customer=u.stripe_customer_id, client_reference_id=u.id,
                  line_items=[{"price": price, "quantity": 1}], allow_promotion_codes=True,
                  subscription_data={"metadata": {"user_id": u.id, "plan": plan}},
                  success_url=f"{C.PUBLIC_URL}/app#/billing?checkout=success", cancel_url=f"{C.PUBLIC_URL}/app#/billing?checkout=cancel")
    if C.STRIPE_AUTOMATIC_TAX:
        params.update(automatic_tax={"enabled": True}, customer_update={"address": "auto", "name": "auto"})
    s = stripe.checkout.Session.create(**params)
    return jsonify({"url": s["url"]})


@bp.post("/api/billing/portal")
@login_required
def portal():
    u = current_user()
    if not u.stripe_customer_id: abort(400, "No billing account yet — choose a plan first.")
    s = _stripe().billing_portal.Session.create(customer=u.stripe_customer_id, return_url=f"{C.PUBLIC_URL}/app#/billing")
    return jsonify({"url": s["url"]})


@bp.post("/api/billing/dev")
@login_required
def dev_activate():
    """Development only: pretend a subscription was paid (or cancelled)."""
    if not C.DEV_MODE: abort(404)
    plan = (request.get_json(force=True, silent=True) or {}).get("plan")
    if plan not in C.PLANS: abort(400)
    u = current_user()
    u.plan = plan; u.sub_status = "" if plan == "free" else "active"
    u.current_period_end = None if plan == "free" else now() + dt.timedelta(days=30)
    Session.commit()
    return jsonify(user_payload(u))


@bp.post("/stripe/webhook")
def webhook():
    import stripe
    if not C.STRIPE_WEBHOOK_SECRET: abort(503)
    try:
        ev = stripe.Webhook.construct_event(request.get_data(), request.headers.get("Stripe-Signature", ""), C.STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        log.warning("bad webhook: %s", e); abort(400)
    ev = ev.to_dict() if hasattr(ev, "to_dict") else dict(ev)
    if Session.get(StripeEvent, ev["id"]): return jsonify({"ok": True, "duplicate": True})
    obj = ev["data"]["object"]; t = ev["type"]
    try:
        if t == "checkout.session.completed" and obj.get("mode") == "subscription":
            u = Session.get(User, obj.get("client_reference_id") or "") or _by_customer(obj.get("customer"))
            if u:
                u.stripe_customer_id = obj.get("customer") or u.stripe_customer_id
                if obj.get("subscription") and C.STRIPE_SECRET_KEY:
                    sync_subscription(u, _stripe().Subscription.retrieve(obj["subscription"]).to_dict())
        elif t in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted",
                   "customer.subscription.paused", "customer.subscription.resumed"):
            u = _by_customer(obj.get("customer")) or Session.get(User, (obj.get("metadata") or {}).get("user_id") or "")
            if u:
                if not u.stripe_customer_id: u.stripe_customer_id = obj.get("customer") or ""
                if t == "customer.subscription.deleted": obj = {**obj, "status": "canceled"}
                if not u.stripe_subscription_id or u.stripe_subscription_id == obj.get("id") or obj.get("status") in ("active", "trialing"):
                    sync_subscription(u, obj)
        elif t == "invoice.payment_failed":
            u = _by_customer(obj.get("customer"))
            if u and u.sub_status in ("active", "trialing"): u.sub_status = "past_due"
        Session.add(StripeEvent(id=ev["id"], type=t)); Session.commit()
    except Exception:
        Session.rollback(); log.exception("webhook %s failed", t); abort(500)  # Stripe retries
    return jsonify({"ok": True})


def _by_customer(cid):
    return Session.scalar(select(User).where(User.stripe_customer_id == cid)) if cid else None


def cancel_now(u):
    """Used on account deletion."""
    if u.stripe_subscription_id and C.STRIPE_SECRET_KEY:
        try: _stripe().Subscription.cancel(u.stripe_subscription_id)
        except Exception: log.exception("could not cancel %s", u.stripe_subscription_id)
