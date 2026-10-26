"""Database models (SQLAlchemy 2). SQLite for local use, PostgreSQL in production via DATABASE_URL."""
import datetime as dt, json, os, uuid
from sqlalchemy import (create_engine, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, func, select, event)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, relationship, scoped_session
from . import config as C


def now(): return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
def new_id(): return uuid.uuid4().hex[:16]


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    pw_hash: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(200), default="")
    specialty: Mapped[str] = mapped_column(String(200), default="Cardiology")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    session_version: Mapped[int] = mapped_column(Integer, default=1)       # bump = log out everywhere
    brand_json: Mapped[str] = mapped_column(Text, default="{}")
    default_language: Mapped[str] = mapped_column(String(8), default="auto")
    default_style: Mapped[str] = mapped_column(String(40), default="modern_educational")
    # billing
    plan: Mapped[str] = mapped_column(String(20), default="free")
    sub_status: Mapped[str] = mapped_column(String(30), default="")
    stripe_customer_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    stripe_subscription_id: Mapped[str] = mapped_column(String(64), default="")
    current_period_end: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, default=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")

    @property
    def brand(self): return json.loads(self.brand_json or "{}")

    def effective_plan(self):
        return self.plan if (self.plan != "free" and self.sub_status in C.ACTIVE_STATUSES) else "free"


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(300), default="Untitled video")
    mode: Mapped[str] = mapped_column(String(20), default="auto_edit")
    language: Mapped[str] = mapped_column(String(8), default="auto")
    style: Mapped[str] = mapped_column(String(40), default="modern_educational")
    script: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft|processing|ready|error
    duration: Mapped[float] = mapped_column(Float, default=0.0)
    counted: Mapped[bool] = mapped_column(Boolean, default=False)      # consumed a monthly video credit
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now, index=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now, onupdate=now)
    exported_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    user = relationship("User", back_populates="projects")

    @property
    def dir(self): return os.path.join(C.STORAGE_DIR, "users", self.user_id, self.id)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    kind: Mapped[str] = mapped_column(String(20))        # generate|preview|final
    payload: Mapped[str] = mapped_column(Text, default="{}")
    priority: Mapped[int] = mapped_column(Integer, default=0)
    state: Mapped[str] = mapped_column(String(20), default="queued", index=True)  # queued|working|done|error|canceled
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="Queued…")
    result: Mapped[str] = mapped_column(Text, default="{}")
    worker: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    heartbeat: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)


class Usage(Base):
    """One row per video credit consumed. Not deleted with the project, so deleting videos
    doesn't hand credits back."""
    __tablename__ = "usage"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    project_id: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now, index=True)


class AIUsage(Base):
    """One row per paid AI action (illustration, animation). Kept when projects are deleted, so the monthly
    allowance can't be reset by deleting videos; a failed action is removed (refunded)."""
    __tablename__ = "ai_usage"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    project_id: Mapped[str] = mapped_column(String(32), index=True)
    kind: Mapped[str] = mapped_column(String(16))
    credits: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now, index=True)


class StripeEvent(Base):
    """Processed webhook event ids — makes webhook handling idempotent."""
    __tablename__ = "stripe_events"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    type: Mapped[str] = mapped_column(String(80))
    received_at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)


_url = C.DATABASE_URL.replace("postgres://", "postgresql://", 1)
if _url.startswith("sqlite:///"): os.makedirs(os.path.dirname(_url[len("sqlite:///"):]) or ".", exist_ok=True)
engine = create_engine(_url, pool_pre_ping=True, **({"connect_args": {"check_same_thread": False, "timeout": 30}} if _url.startswith("sqlite") else {}))
if _url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(con, _):
        cur = con.cursor(); cur.execute("PRAGMA journal_mode=WAL"); cur.execute("PRAGMA foreign_keys=ON"); cur.close()
Session = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))


def init_db():
    Base.metadata.create_all(engine)


def month_start():
    n = now(); return n.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def videos_used(db, user):
    """Videos created this calendar month (UTC) that consumed a credit."""
    return db.scalar(select(func.count(Usage.id)).where(Usage.user_id == user.id, Usage.created_at >= month_start())) or 0


def ai_credits_used(db, user):
    return db.scalar(select(func.coalesce(func.sum(AIUsage.credits), 0)).where(AIUsage.user_id == user.id, AIUsage.created_at >= month_start())) or 0


def ai_credits_left(db, user):
    return max(0, C.PLANS[user.effective_plan()]["ai_credits"] - ai_credits_used(db, user))


def ai_count(db, project_id, kind):
    """Paid AI actions of this kind ever made for a project (the per-video ceiling)."""
    return db.scalar(select(func.count(AIUsage.id)).where(AIUsage.project_id == project_id, AIUsage.kind == kind)) or 0


def ai_charge(db, user, project_id, kind, n=1):
    """Record n actions; returns the row ids (to refund on failure)."""
    rows = [AIUsage(user_id=user.id, project_id=project_id, kind=kind, credits=C.AI_COST[kind]) for _ in range(n)]
    db.add_all(rows); db.commit()
    return [r.id for r in rows]


def ai_refund(db, ids):
    if ids: db.execute(AIUsage.__table__.delete().where(AIUsage.id.in_(list(ids)))); db.commit()


def ai_status(db, user, project_id=None):
    out = {"left": ai_credits_left(db, user), "monthly": C.PLANS[user.effective_plan()]["ai_credits"], "cost": C.AI_COST, "per_video": C.AI_PER_VIDEO}
    if project_id: out["video"] = {k: ai_count(db, project_id, k) for k in C.AI_PER_VIDEO}
    return out
