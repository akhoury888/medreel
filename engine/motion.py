"""AI-designed medical animations ("gen/<id>").

Claude (Sonnet) never writes code. It writes a *scene description* — JSON using a small, fixed vocabulary of
drawing pieces (vessels/tubes with narrowings and flowing cells, organ-shaped blobs, glowing shapes, arrows,
labels, counters, pulses) with keyframed motion. This module validates that description (unknown pieces and
out-of-range numbers are dropped/clamped) and draws it frame by frame in MedReel's house style, exactly like
the hand-made animations. Safe by construction: nothing from the model is ever executed.

Stage: 1000 x 1000 design units, fitted into whatever panel the layout gives (split, side card, full screen).
Time: every "show"/"anim" time is a fraction 0..1 of the scene's duration on the timeline.
"""
import functools, json, math, os, re, uuid, logging
import numpy as np, cv2
from .draw import font, fit_font, text_sprite, blit, rounded_mask, ease_out_cubic, ease_in_out, ease_out_back, clamp, draw_arrow
from . import graphics as G

log = logging.getLogger("medreel.motion")
GEN_DIR = None  # set by the renderer to <project>/anims

# ------------------------------------------------------------------ palette (names Claude may use)
PALETTE = {
    "blood": "#d8263f", "blood_dark": "#8e1026", "cell": "#e8384f", "vessel_wall": "#f0a2a8", "lumen": "#5a0d1c",
    "plaque": "#f2c14e", "fat": "#f6d77a", "calcium": "#f4f1e6", "clot": "#7a0f1f", "tissue": "#e98a8f", "muscle": "#c8495a",
    "bone": "#efe6d2", "cartilage": "#bfe3f0", "air": "#9fd8ff", "airway": "#f3b8b0", "lung": "#f19aa6", "nerve": "#ffd36e",
    "brain": "#f4a8b8", "skin": "#f2c4a4", "liver": "#9b3a3a", "bile": "#9ccf4c", "kidney": "#b4485a", "urine": "#f5d76e",
    "stomach": "#f0a090", "acid": "#c7f05a", "virus": "#7ee08a", "bacteria": "#b98cff", "immune": "#f5f5ff", "insulin": "#5ad1ff",
    "sugar": "#ffffff", "oxygen": "#6fd3ff", "co2": "#a6a6c8", "drug": "#3ee0c0", "device": "#d9e2ef", "metal": "#c3ccd8",
    "good": "#3ddc84", "bad": "#ff5a5a", "warning": "#ffb020", "white": "#ffffff", "muted": "#9fb0cc", "dark": "#0b1730",
}


def _rgb(c, accent=(46, 160, 255)):
    if isinstance(c, str):
        c = c.strip().lower()
        if c == "accent": return tuple(int(x) for x in accent)
        c = PALETTE.get(c, c)
        m = re.fullmatch(r"#?([0-9a-f]{6})", c)
        if m: h = m.group(1); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (255, 255, 255)


# ------------------------------------------------------------------ validation (the only gate between the model and the renderer)
TYPES = {"tube", "particles", "shape", "path", "arrow", "label", "counter", "pulse"}
NUM = {  # prop -> (min, max)
    "x": (-200, 1200), "y": (-200, 1200), "r": (0, 700), "rx": (0, 700), "ry": (0, 700), "w": (0, 1400), "h": (0, 1400),
    "width": (1, 400), "opacity": (0, 1), "draw": (0, 1), "scale": (0, 4), "rot": (-720, 720), "speed": (-3, 3),
    "amount": (0, 1), "glow": (0, 1), "value": (-1e6, 1e6), "size": (10, 140), "spread": (0, 1), "at": (0, 1), "length": (0.02, 1),
}
MAX_ELEMS, MAX_PTS, MAX_PARTICLES = 40, 48, 160


def _num(v, k, d=None):
    try: v = float(v)
    except (TypeError, ValueError): return d
    if not math.isfinite(v): return d
    lo, hi = NUM.get(k, (-1e6, 1e6)); return clamp(v, lo, hi)


def _pts(v, closed_ok=True):
    out = []
    for p in (v or [])[:MAX_PTS]:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = _num(p[0], "x"), _num(p[1], "y")
            if x is not None and y is not None: out.append((x, y))
    return out


def _txt(v, n=48):
    s = re.sub(r"[\x00-\x1f<>{}]", " ", str(v or "")).strip()
    return s[:n]


def validate(scene):
    """Untrusted JSON -> a clean scene dict the renderer understands (or None)."""
    if not isinstance(scene, dict): return None
    out = {"title": _txt(scene.get("title"), 40), "bg": scene.get("bg") if scene.get("bg") in ("dark", "deep", "teal", "purple") else "dark",
           "elements": []}
    for e in (scene.get("elements") or [])[:MAX_ELEMS]:
        if not isinstance(e, dict) or e.get("type") not in TYPES: continue
        c = {"type": e["type"], "id": _txt(e.get("id"), 20)}
        for k in NUM:
            if k in e and not isinstance(e[k], (list, dict)):
                v = _num(e[k], k)
                if v is not None: c[k] = v
        for k in ("color", "fill", "outline", "wall", "lumen", "plaque", "text_color"):
            if isinstance(e.get(k), str): c[k] = e[k][:24]
        for k in ("points", "from", "to", "point_to"):
            if k in e:
                p = _pts(e[k] if k == "points" else [e[k]])
                if p: c[k] = p if k == "points" else p[0]
        if e["type"] == "shape": c["kind"] = e.get("kind") if e.get("kind") in ("circle", "ellipse", "rect", "blob") else "ellipse"
        if e["type"] == "particles":
            c["style"] = e.get("style") if e.get("style") in ("cell", "dot", "spark", "ring", "virus") else "dot"
            c["count"] = int(clamp(_num(e.get("count"), "value", 20) or 20, 1, 80))
            if isinstance(e.get("along"), str): c["along"] = _txt(e["along"], 20)
            if isinstance(e.get("area"), (list, tuple)) and len(e["area"]) == 4:
                a = [_num(v, "x", 0) for v in e["area"]]; c["area"] = a
        if e["type"] == "tube" and isinstance(e.get("narrow"), dict):
            n = e["narrow"]; c["narrow"] = {"at": _num(n.get("at"), "at", 0.5), "length": _num(n.get("length"), "length", 0.25)}
        if e["type"] in ("label", "counter"):
            c["text"] = _txt(e.get("text"), 48); c["suffix"] = _txt(e.get("suffix"), 6); c["prefix"] = _txt(e.get("prefix"), 6)
            c["pill"] = bool(e.get("pill")); c["bold"] = e.get("bold", True) is not False
        if e["type"] in ("path", "shape"): c["closed"] = bool(e.get("closed", e["type"] == "shape")); c["dashed"] = bool(e.get("dashed"))
        sh = e.get("show")
        if isinstance(sh, (list, tuple)) and len(sh) == 2:
            a, b = _num(sh[0], "at", 0), _num(sh[1], "at", 1); c["show"] = (min(a, b), max(a, b))
        c["enter"] = e.get("enter") if e.get("enter") in ("fade", "pop", "grow", "draw", "none") else "fade"
        an = {}
        for k, keys in (e.get("anim") or {}).items() if isinstance(e.get("anim"), dict) else []:
            if k not in NUM or not isinstance(keys, list): continue
            ks = []
            for kv in keys[:12]:
                if isinstance(kv, (list, tuple)) and len(kv) == 2:
                    t, v = _num(kv[0], "at"), _num(kv[1], k)
                    if t is not None and v is not None: ks.append((t, v))
            if ks: an[k] = sorted(ks)
        if an: c["anim"] = an
        # minimum geometry per type
        if c["type"] in ("tube", "path") and len(c.get("points", [])) < 2: continue
        if c["type"] == "shape" and c["kind"] == "blob" and len(c.get("points", [])) < 3: continue
        if c["type"] == "arrow" and not ("from" in c and "to" in c): continue
        if c["type"] == "label" and not c["text"]: continue
        out["elements"].append(c)
    n_part = 0
    for c in out["elements"]:
        if c["type"] == "particles":
            c["count"] = min(c["count"], max(0, MAX_PARTICLES - n_part)); n_part += c["count"]
    return out if out["elements"] else None


# ------------------------------------------------------------------ helpers
def _val(e, k, u, d):
    """Animated numeric property at scene time u (0..1): keyframes with smooth easing, else the static value."""
    ks = e.get("anim", {}).get(k)
    if not ks: return e.get(k, d)
    if u <= ks[0][0]: return ks[0][1]
    for (t0, v0), (t1, v1) in zip(ks, ks[1:]):
        if u <= t1: return v0 + (v1 - v0) * ease_in_out((u - t0) / max(1e-6, t1 - t0))
    return ks[-1][1]


def _integral(e, k, u, d, n=24):
    """∫0..u of an animated property (positions of particles whose speed changes move smoothly)."""
    if "anim" not in e or k not in e["anim"]: return e.get(k, d) * u
    xs = np.linspace(0, u, n + 1); ys = [_val(e, k, x, d) for x in xs]
    return float(sum((ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i]) / 2 for i in range(n)))


def _vis(e, u):
    """Visibility 0..1 and the entrance progress (for pop/grow/draw)."""
    a, b = e.get("show", (0, 1)); fd = 0.06
    if u < a or u > b + fd: return 0.0, 0.0
    enter = clamp((u - a) / fd, 0, 1) if a > 0 else 1.0
    leave = clamp((b + fd - u) / fd, 0, 1) if b < 1 else 1.0
    return min(enter, leave), clamp((u - a) / 0.14, 0, 1) if a > 0 else 1.0


def _spline(pts, n=120, closed=False):
    """Catmull-Rom through the points -> dense polyline (smooth organ outlines and vessels)."""
    P = np.array(pts, np.float32)
    if len(P) < 3: return np.linspace(P[0], P[-1], n) if len(P) == 2 else P
    if closed: P = np.vstack([P[-1:], P, P[:2]])
    else: P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    segs = len(P) - 3; per = max(4, n // segs); out = []
    for i in range(segs):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        t = np.linspace(0, 1, per, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2:-1] if not closed else P[1:2])
    return np.vstack(out)


def _arc(poly):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    return d / max(d[-1], 1e-6), d[-1]


def _normals(poly):
    tg = np.gradient(poly, axis=0); tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-6
    return np.stack([-tg[:, 1], tg[:, 0]], 1)


def _shade(c, k): return tuple(int(clamp(v * k, 0, 255)) for v in c)
def _mix(a, b, t): return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


BG = {"dark": ((14, 30, 60), (5, 12, 28)), "deep": ((10, 16, 34), (2, 4, 12)), "teal": ((8, 46, 58), (3, 16, 24)), "purple": ((34, 18, 58), (10, 6, 24))}


@functools.lru_cache(maxsize=8)
def _bg(w, h, kind):
    top, bot = BG.get(kind, BG["dark"])
    return G._bg(w, h, top, bot) if hasattr(G, "_bg") else np.zeros((h, w, 3), np.uint8)


# ------------------------------------------------------------------ element painters (half-resolution layer)
class _Ctx:
    def __init__(self, img, glow, k, ox, oy, accent, u, t, D):
        self.img, self.glow, self.k, self.ox, self.oy, self.accent, self.u, self.t, self.D = img, glow, k, ox, oy, accent, u, t, D
        self.tubes = {}

    def P(self, x, y): return (self.ox + x * self.k, self.oy + y * self.k)
    def S(self, v): return v * self.k


def _layer(ctx, alpha, fn):
    """Draw fn() on a copy and blend with the element's opacity."""
    if alpha >= 0.99: fn(ctx.img); return
    ov = ctx.img.copy(); fn(ov); cv2.addWeighted(ov, alpha, ctx.img, 1 - alpha, 0, ctx.img)


def _tube(ctx, e, a, grow, geometry_only=False):
    u = ctx.u
    poly = _spline(e["points"], 160); s, _ = _arc(poly)
    pts = np.array([ctx.P(x, y) for x, y in poly], np.float32)
    w = ctx.S(_val(e, "width", u, 60)) / 2
    prof = np.ones(len(pts), np.float32)
    if "narrow" in e:
        at, ln = e["narrow"]["at"], e["narrow"]["length"]
        amt = _val(e, "amount", u, 0.0)
        bump = np.clip(1 - ((s - at) / (ln / 2)) ** 2, 0, 1) ** 1.5
        prof = 1 - amt * bump
    reach = len(pts) if grow >= 1 or e.get("enter") != "draw" else max(2, int(len(pts) * grow))
    if geometry_only:
        ctx.tubes[e.get("id") or f"tube{len(ctx.tubes)}"] = (pts, s, w * prof * 0.86); return
    wall, lumen = _rgb(e.get("wall", "vessel_wall"), ctx.accent), _rgb(e.get("lumen", "lumen"), ctx.accent)
    plaque = _rgb(e.get("plaque", "plaque"), ctx.accent)

    def paint(img):
        wt = max(2, int(w * 0.16))
        for i in range(reach):  # outer wall with soft shading
            cv2.circle(img, (int(pts[i][0]), int(pts[i][1])), int(w + wt), _shade(wall, 0.72), -1, cv2.LINE_AA)
        for i in range(reach):
            cv2.circle(img, (int(pts[i][0]), int(pts[i][1])), int(w + wt * 0.45), wall, -1, cv2.LINE_AA)
        if "narrow" in e:
            for i in range(reach):
                if prof[i] < 0.995: cv2.circle(img, (int(pts[i][0]), int(pts[i][1])), int(w * 1.0), plaque, -1, cv2.LINE_AA)
        for i in range(reach):
            cv2.circle(img, (int(pts[i][0]), int(pts[i][1])), max(1, int(w * prof[i] * 0.86)), lumen, -1, cv2.LINE_AA)
        core = _mix(lumen, (255, 255, 255), 0.16); ov = img.copy()  # depth: lighter core down the middle of the lumen
        for i in range(reach): cv2.circle(ov, (int(pts[i][0]), int(pts[i][1])), max(1, int(w * prof[i] * 0.42)), core, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
        nr = _normals(pts[:reach])
        hl = (pts[:reach] - nr * (w * 0.62)).astype(np.int32)  # specular stripe along the wall
        cv2.polylines(img, [hl], False, _mix(wall, (255, 255, 255), 0.55), max(1, int(w * 0.10)), cv2.LINE_AA)
    _layer(ctx, a, paint)
    gl = [v * 0.22 * a for v in wall]  # soft halo around the tube
    for i in range(0, reach, 3): cv2.circle(ctx.glow, (int(pts[i][0]), int(pts[i][1])), int(w * 1.25), gl, -1)
    ctx.tubes[e.get("id") or f"tube{len(ctx.tubes)}"] = (pts, s, w * prof * 0.86)


_PH = np.random.default_rng(11).random((MAX_PARTICLES * 2, 4))


def _particles(ctx, e, a, grow, idx):
    u = ctx.u; n = e["count"]; col = _rgb(e.get("color", "cell" if e["style"] == "cell" else "white"), ctx.accent)
    size = ctx.S(e.get("size", 22 if e["style"] == "cell" else 12))
    ph = _PH[(idx * 37) % MAX_PARTICLES:][:n]
    if len(ph) < n: ph = np.vstack([ph, _PH[: n - len(ph)]])
    travel = _integral(e, "speed", u, 1.0) * ctx.D / 6.0  # speed 1 = crosses the path once in ~6 s
    tube = ctx.tubes.get(e.get("along", "")) if e.get("along") else None
    out = []
    if tube is not None:
        pts, s, rad = tube
        for i in range(n):
            pos = (ph[i, 0] + travel * (0.85 + 0.3 * ph[i, 2])) % 1.0
            j = int(np.searchsorted(s, pos)); j = min(len(pts) - 1, j)
            nr = _normals(pts[max(0, j - 1): j + 2])[0] if len(pts) > 2 else np.array([0, 1])
            lat = (ph[i, 1] * 2 - 1) * 0.7 * rad[j]
            out.append((pts[j][0] + nr[0] * lat, pts[j][1] + nr[1] * lat, ph[i, 3]))
    elif "points" in e:
        poly = _spline(e["points"], 120, closed=bool(e.get("closed"))); s, _ = _arc(poly)
        pts = np.array([ctx.P(x, y) for x, y in poly], np.float32); spread = ctx.S(_val(e, "spread", u, 0.0) * 120)
        for i in range(n):
            pos = (ph[i, 0] + travel * (0.85 + 0.3 * ph[i, 2])) % 1.0
            j = min(len(pts) - 1, int(np.searchsorted(s, pos)))
            out.append((pts[j][0] + (ph[i, 1] - .5) * spread, pts[j][1] + (ph[i, 3] - .5) * spread, ph[i, 3]))
    else:  # drifting in an area (x0, y0, x1, y1)
        x0, y0, x1, y1 = e.get("area", [200, 200, 800, 800])
        for i in range(n):
            ang = ph[i, 2] * 6.283 + travel * 2.2
            x = x0 + (x1 - x0) * ((ph[i, 0] + 0.08 * math.sin(ang + i)) % 1.0); y = y0 + (y1 - y0) * ((ph[i, 1] + 0.08 * math.cos(ang * 0.8 + i)) % 1.0)
            out.append((*ctx.P(x, y), ph[i, 3]))
    k = ease_out_back(grow) if e.get("enter") == "pop" else 1.0

    def paint(img):
        for x, y, j in out:
            r = max(1, int(size * (0.8 + 0.4 * j) * k / 2))
            if e["style"] == "cell":
                cv2.circle(img, (int(x), int(y)), r, col, -1, cv2.LINE_AA)
                cv2.circle(img, (int(x), int(y)), max(1, int(r * 0.5)), _shade(col, 0.72), -1, cv2.LINE_AA)
            elif e["style"] == "ring":
                cv2.circle(img, (int(x), int(y)), r, col, max(1, r // 3), cv2.LINE_AA)
            elif e["style"] == "virus":
                cv2.circle(img, (int(x), int(y)), r, col, -1, cv2.LINE_AA)
                for q in range(8):
                    an = q * 0.785 + j * 3
                    cv2.circle(img, (int(x + math.cos(an) * r * 1.35), int(y + math.sin(an) * r * 1.35)), max(1, r // 4), _shade(col, 1.2), -1, cv2.LINE_AA)
            else:
                cv2.circle(img, (int(x), int(y)), r, col, -1, cv2.LINE_AA)
        if e["style"] in ("spark", "dot") or e.get("glow", 0) > 0:
            for x, y, j in out: cv2.circle(ctx.glow, (int(x), int(y)), max(1, int(size * 0.9)), [v * (e.get("glow", 0.6)) for v in col], -1)
    _layer(ctx, a, paint)


def _shape(ctx, e, a, grow):
    u = ctx.u; kind = e["kind"]
    sc = _val(e, "scale", u, 1.0) * (ease_out_back(grow) if e.get("enter") == "pop" else ease_out_cubic(grow) if e.get("enter") == "grow" else 1.0)
    fill = _rgb(e.get("fill", e.get("color", "accent")), ctx.accent) if e.get("fill", "x") != "none" else None
    outline = _rgb(e["outline"], ctx.accent) if e.get("outline") and e["outline"] != "none" else None
    x, y = ctx.P(_val(e, "x", u, 500), _val(e, "y", u, 500)); rot = _val(e, "rot", u, 0)
    if kind == "blob":
        cx = np.mean([p[0] for p in e["points"]]); cy = np.mean([p[1] for p in e["points"]])
        dx, dy = _val(e, "x", u, cx) - cx, _val(e, "y", u, cy) - cy
        poly = _spline(e["points"], 140, closed=True)
        ca, sa = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        P = [ctx.P(cx + dx + ((px - cx) * ca - (py - cy) * sa) * sc, cy + dy + ((px - cx) * sa + (py - cy) * ca) * sc) for px, py in poly]
        cnt = np.array(P, np.int32); x, y = ctx.P(cx + dx, cy + dy)
        rx = ry = max(4, ctx.S(max(np.ptp([p[0] for p in poly]), np.ptp([p[1] for p in poly])) / 2 * sc))
    else:
        if kind == "circle": rx = ry = ctx.S(_val(e, "r", u, 60)) * sc
        elif kind == "rect": rx, ry = ctx.S(_val(e, "w", u, 120)) / 2 * sc, ctx.S(_val(e, "h", u, 80)) / 2 * sc
        else: rx, ry = ctx.S(_val(e, "rx", u, 80)) * sc, ctx.S(_val(e, "ry", u, 50)) * sc
        if kind == "rect":
            box = cv2.boxPoints(((x, y), (2 * rx, 2 * ry), rot)); cnt = box.astype(np.int32)
        else:
            cnt = cv2.ellipse2Poly((int(x), int(y)), (max(1, int(rx)), max(1, int(ry))), int(rot), 0, 360, 4)
    if rx < 1 or ry < 1: return
    width = max(1, int(ctx.S(_val(e, "width", u, 4))))

    def paint(img):
        if fill is not None:  # smooth 3D shading: darker towards the edge, soft highlight top-left
            H_, W_ = img.shape[:2]
            x0, y0 = max(0, cnt[:, 0].min() - 2), max(0, cnt[:, 1].min() - 2)
            x1, y1 = min(W_, cnt[:, 0].max() + 3), min(H_, cnt[:, 1].max() + 3)
            if x1 - x0 > 2 and y1 - y0 > 2:
                m = np.zeros((y1 - y0, x1 - x0), np.uint8); cv2.fillPoly(m, [cnt - [x0, y0]], 255, cv2.LINE_AA)
                dt = cv2.distanceTransform((m > 127).astype(np.uint8), cv2.DIST_L2, 3); dt /= max(dt.max(), 1e-6)
                yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
                hx, hy = x - rx * 0.32, y - ry * 0.38
                hl = np.exp(-(((xx - hx) / max(rx * 0.55, 1)) ** 2 + ((yy - hy) / max(ry * 0.45, 1)) ** 2))
                base = np.array(fill, np.float32)
                col = base * (0.55 + 0.45 * dt[..., None] ** 0.55) + (255 - base) * (0.42 * hl[..., None])
                al = (m.astype(np.float32) / 255)[..., None]
                roi = img[y0:y1, x0:x1].astype(np.float32)
                img[y0:y1, x0:x1] = np.clip(roi * (1 - al) + col * al, 0, 255).astype(np.uint8)
        if outline is not None:
            cv2.polylines(img, [cnt], True, outline, width, cv2.LINE_AA)
        if e.get("glow", 0) > 0:
            cv2.fillPoly(ctx.glow, [cnt], [v * e["glow"] for v in (fill or outline or (255, 255, 255))])
    _layer(ctx, a * _val(e, "opacity", u, 1.0), paint)


def _path(ctx, e, a, grow):
    u = ctx.u; poly = _spline(e["points"], 140, closed=e.get("closed", False))
    pts = np.array([ctx.P(x, y) for x, y in poly], np.int32)
    dr = _val(e, "draw", u, 1.0) * (grow if e.get("enter") == "draw" else 1.0)
    pts = pts[: max(2, int(len(pts) * dr))]
    col = _rgb(e.get("color", "white"), ctx.accent); width = max(1, int(ctx.S(_val(e, "width", u, 6))))

    def paint(img):
        if e.get("dashed"):
            for i in range(0, len(pts) - 1, 6): cv2.polylines(img, [pts[i:i + 4]], False, col, width, cv2.LINE_AA)
        else:
            cv2.polylines(img, [pts], bool(e.get("closed")) and dr >= 1, col, width, cv2.LINE_AA)
        if e.get("glow", 0) > 0: cv2.polylines(ctx.glow, [pts], False, [v * e["glow"] for v in col], width * 3)
    _layer(ctx, a * _val(e, "opacity", u, 1.0), paint)


def _pulse(ctx, e, a):
    u = ctx.u; x, y = ctx.P(_val(e, "x", u, 500), _val(e, "y", u, 500)); r = ctx.S(_val(e, "r", u, 60))
    col = _rgb(e.get("color", "accent"), ctx.accent); ph = (ctx.t * 0.9) % 1.0

    def paint(img):
        for q in (0, 0.5):  # rings grow and fade out
            p = (ph + q) % 1.0
            ov = img.copy(); cv2.circle(ov, (int(x), int(y)), int(r * (0.6 + 1.2 * p)), col, max(2, int(r * 0.07)), cv2.LINE_AA)
            cv2.addWeighted(ov, 0.85 * (1 - p), img, 1 - 0.85 * (1 - p), 0, img)
        cv2.circle(ctx.glow, (int(x), int(y)), int(r * 0.7), [v * 0.5 for v in col], -1)
    _layer(ctx, a, paint)


# ------------------------------------------------------------------ full-resolution overlays (crisp text and arrows)
def _overlays(img, scene, elems, u, t, accent, k, ox, oy, s, lang):
    P = lambda x, y: (ox + x * k, oy + y * k)
    for e in elems:
        a, grow = _vis(e, u)
        if a <= 0.01: continue
        if e["type"] == "arrow":
            p0, p1 = P(*e["from"]), P(*e["to"]); col = _rgb(e.get("color", "white"), accent)
            dr = _val(e, "draw", u, 1.0) * (ease_out_cubic(grow) if e.get("enter") in ("draw", "fade") else 1.0)
            ov = img.copy(); draw_arrow(ov, p0, p1, col, max(2, int(k * _val(e, "width", u, 6))), progress=dr, head=int(max(10, k * 26)))
            cv2.addWeighted(ov, a, img, 1 - a, 0, img)
        elif e["type"] in ("label", "counter"):
            if e["type"] == "counter":
                v = _val(e, "value", u, 0); txt = f"{e.get('prefix', '')}{int(round(v))}{e.get('suffix', '')}"
                if e.get("text"): txt = f"{txt} {e['text']}"
            else: txt = e["text"]
            size = max(14, int(k * e.get("size", 40 if e["type"] == "label" else 84)))
            x, y = P(_val(e, "x", u, 500), _val(e, "y", u, 500))
            fnt = fit_font(txt, "Poppins", 700 if e.get("bold", True) else 500, size, img.shape[1] * 0.86)
            col = _rgb(e.get("text_color", e.get("color", "white")), accent)
            spr = text_sprite(txt, fnt, (255, 255, 255) if e.get("pill") else col, shadow=0 if e.get("pill") else max(2, int(4 * s)), pad=0 if e.get("pill") else 8)
            wv, hv = spr.shape[1] + (size if e.get("pill") else 0), spr.shape[0] + (int(size * .7) if e.get("pill") else 0)
            bx, by = clamp(x - wv / 2, 12 * s, img.shape[1] - wv - 12 * s), clamp(y - hv / 2, 12 * s, img.shape[0] - hv - 12 * s)
            pop = ease_out_back(grow) if e.get("enter") == "pop" else 1.0
            dy = (1 - ease_out_cubic(grow)) * 24 * s if e.get("enter") == "fade" else 0
            if e.get("point_to"):
                tx, ty = P(*e["point_to"]); ax, ay = bx + wv / 2, (by + hv if ty > by + hv else by)
                ov = img.copy(); cv2.line(ov, (int(ax), int(ay + dy)), (int(tx), int(ty)), (255, 255, 255), max(2, int(3 * s)), cv2.LINE_AA)
                cv2.circle(ov, (int(tx), int(ty)), max(3, int(7 * s)), (255, 255, 255), -1, cv2.LINE_AA)
                cv2.addWeighted(ov, a, img, 1 - a, 0, img)
            if e.get("pill"):
                card = np.zeros((hv, wv, 4), np.uint8); card[..., :3] = _rgb(e.get("color", "accent"), accent); card[..., 3] = rounded_mask(wv, hv, hv // 2)
                if pop != 1.0:
                    from .draw import scale_rgba
                    card = scale_rgba(card, max(0.05, pop)); spr2 = scale_rgba(spr, max(0.05, pop))
                    blit(img, card, bx + (wv - card.shape[1]) / 2, by + (hv - card.shape[0]) / 2 + dy, a * 0.95)
                    blit(img, spr2, bx + (wv - spr2.shape[1]) / 2, by + (hv - spr2.shape[0]) / 2 + dy, a)
                else:
                    blit(img, card, bx, by + dy, a * 0.95); blit(img, spr, bx + size / 2, by + (hv - spr.shape[0]) / 2 + dy, a)
            else:
                blit(img, spr, bx, by + dy, a)


# ------------------------------------------------------------------ public renderer (asset "gen")
@functools.lru_cache(maxsize=32)
def _load(path, mtime):
    try: return validate(json.load(open(path)))
    except Exception: log.exception("bad animation file %s", path); return None


def load(scene_id):
    if not GEN_DIR or not re.fullmatch(r"[0-9a-f]{12}", scene_id or ""): return None
    p = os.path.join(GEN_DIR, scene_id + ".json")
    return _load(p, os.path.getmtime(p)) if os.path.exists(p) else None


def render_scene(scene, W, H, t, D, accent=(46, 160, 255), lang="en", title=None):
    u = clamp(t / max(D, 0.1), 0, 1)
    f = 1.0; w, h = max(64, int(W * f)), max(64, int(H * f))  # full resolution: sharp edges, no upscaling blur
    img = _bg(w, h, scene.get("bg", "dark")).copy()
    glow = np.zeros((h, w, 3), np.float32)
    portrait = H > W * 1.3
    side = min(w, h) * (0.98 if not portrait else 1.0)
    k = side / 1000.0; ox, oy = (w - side) / 2, (h - side) / 2 + (h * 0.04 if portrait else h * 0.03)
    ctx = _Ctx(img, glow, k, ox, oy, accent, u, t, D)
    elems = scene["elements"]
    for e in elems:  # pre-pass: tube geometry, so particles can flow in a tube drawn later (or earlier)
        if e["type"] == "tube" and _vis(e, u)[0] > 0.01: _tube(ctx, e, 0.0, 1.0, geometry_only=True)
    for i, e in enumerate(elems):  # then paint in the designer's order (later = on top)
        a, grow = _vis(e, u)
        if a <= 0.01: continue
        if e["type"] == "tube": _tube(ctx, e, a, grow)
        elif e["type"] == "particles": _particles(ctx, e, a, grow, i)
        elif e["type"] == "shape": _shape(ctx, e, a, grow)
        elif e["type"] == "path": _path(ctx, e, a, grow)
        elif e["type"] == "pulse": _pulse(ctx, e, a)
    if glow.max() > 0:
        q = 4  # glow is soft by nature: blur it at quarter size (fast), then bring it back up
        gs = cv2.resize(glow, (max(8, w // q), max(8, h // q)), interpolation=cv2.INTER_AREA)
        gs = cv2.GaussianBlur(gs, (0, 0), max(1.5, w * 0.018 / q))
        g = cv2.resize(gs, (w, h), interpolation=cv2.INTER_LINEAR)
        img = np.clip(img.astype(np.float32) + g * 0.9, 0, 255).astype(np.uint8)
    out = img if (w, h) == (W, H) else cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
    s = max(W / 1080, 0.8 * G.UI_SCALE)
    K = k / f
    _overlays(out, scene, [e for e in elems if e["type"] in ("arrow", "label", "counter")], u, t, accent, K, ox / f, oy / f, s, lang)
    ttl = title or scene.get("title")
    if ttl:
        chip_y = int(max(0.09 * H, 175 * W / 1080) if portrait else 0.06 * H + (175 * W / 1080 if W > 900 else 0))
        G._chip(out, ttl, int(40 * W / 1080), chip_y, accent, max(16, int(36 * s)), rtl=lang == "ar")
    return out


def gen(W, H, t, D, mode="", accent=(46, 160, 255), title=None, lang="en", **_):
    sc = load(mode)
    if sc is None:
        img = np.zeros((H, W, 3), np.uint8); img[:] = (14, 22, 40); return img
    return render_scene(sc, W, H, t, D, accent, lang)


G.ASSETS["gen"] = gen


# ------------------------------------------------------------------ Claude: design a scene
SPEC = """You design short medical explainer animations for a doctor's vertical social-media video.
You do NOT write code. You output ONE JSON scene that a renderer draws. Stage: 1000 x 1000 units (x right, y down),
keep everything inside 60..940. All times are FRACTIONS of the scene length (0 = start, 1 = end); the scene lasts 5-8 s.

Scene: {"title": "<2-4 words in the video language>", "bg": "dark|deep|teal|purple", "elements": [ ... up to 30 ... ]}

Every element may have: "id", "show": [t0, t1] (when it is visible), "enter": "fade|pop|grow|draw",
"anim": {"<prop>": [[t, value], [t, value], ...]} (smoothly eased keyframes for any numeric prop).
Colours: "#rrggbb", "accent" (the doctor's brand colour) or a palette name:
""" + ", ".join(PALETTE) + """

Pieces:
- tube: a vessel, airway, bronchus, intestine, nerve, duct, ureter... {"type":"tube","id":"v1","points":[[x,y],...3-8 pts, smooth curve],
  "width": 60-220, "wall": colour, "lumen": colour, "plaque": colour, "narrow": {"at": 0..1 along the tube, "length": 0.1-0.5},
  "amount": 0..0.95 (how narrowed; animate it: "anim":{"amount":[[0.1,0],[0.6,0.7]]})}
- particles: things that flow or float. {"type":"particles","style":"cell|dot|spark|ring|virus","count":5-60,"size":6-40,"color":...,
  "speed": -2..2 (1 = crosses once in 6 s; animate it to show flow slowing/stopping/restoring),
  and EITHER "along":"<tube id>" (flows inside that tube) OR "points":[[x,y],...] (follows a path) OR "area":[x0,y0,x1,y1] (drifts)}
- shape: {"type":"shape","kind":"circle|ellipse|rect|blob","x","y","r" | "rx","ry" | "w","h","rot","fill":colour|"none","outline":colour,
  "width","glow":0..1,"opacity","scale"}. blob = organic outline from "points" (6-20 points around the organ; smooth closed curve):
  use blobs for organs (lung, kidney, stomach, liver, brain, heart, uterus, tumour, cyst, cell). Shapes get automatic 3D shading.
- path: a line or outline. {"type":"path","points":[...],"color","width","closed":bool,"dashed":bool,"glow","draw":0..1 (animate to draw it on)}
- arrow: {"type":"arrow","from":[x,y],"to":[x,y],"color","width"} (animates drawing on when it appears)
- label: short text in the video language. {"type":"label","text":"<1-4 words>","x","y","size":28-60,"pill":true|false,
  "color" (pill background or text colour),"point_to":[x,y] (draws a pointer line to that spot)}
- counter: an animated number. {"type":"counter","x","y","size":60-120,"anim":{"value":[[0.2,0],[0.7,75]]},"suffix":"%","text":"<optional word>"}
- pulse: pulsing rings to draw attention. {"type":"pulse","x","y","r","color"}

Rules:
1. Tell a mini-story in 2-4 beats that matches EXACTLY what the doctor says (e.g. normal -> problem appears -> treatment -> result).
   Use "show" and "anim" so things change over time; nothing should be static for the whole scene.
2. Medically accurate and simple; recognisable anatomy built from blobs/tubes, big and centred (main subject should fill ~60% of the stage).
3. At most 3 labels visible at the same time, 1-4 words each, in the SAME language as the doctor's sentence (Arabic stays Arabic).
   Labels must not overlap each other or cover the main subject; put them in empty space and use point_to.
4. Leave the top 140 units free (a title chip sits there). No gore, no frightening imagery, no real people, no brand names.
5. Make it look premium: 12-25 elements; a large clear main subject; supporting context drawn behind it (faint shapes with
   opacity 0.25-0.5, e.g. surrounding tissue, a body outline, cells); glow on the important thing; colours from the palette;
   smooth motion with several keyframes (things grow, flow, pulse, change colour) rather than objects just appearing.
6. Return ONLY the compact JSON object (no spaces needed), no explanation, no markdown."""

EXAMPLE = {"title": "Asthma attack", "bg": "dark", "elements": [
    {"type": "tube", "id": "air", "points": [[110, 560], [330, 520], [560, 560], [880, 500]], "width": 190, "wall": "airway", "lumen": "#3a1424",
     "plaque": "#f7c9c0", "narrow": {"at": 0.5, "length": 0.55}, "anim": {"amount": [[0.15, 0.0], [0.45, 0.62], [0.72, 0.62], [0.9, 0.1]]}},
    {"type": "particles", "style": "dot", "along": "air", "count": 26, "size": 12, "color": "air", "glow": 0.6,
     "anim": {"speed": [[0, 1.2], [0.4, 0.25], [0.72, 0.25], [0.9, 1.3]]}},
    {"type": "shape", "kind": "blob", "points": [[470, 505], [530, 495], [585, 512], [560, 540], [495, 538]], "fill": "#f7e3a8", "show": [0.3, 0.78], "enter": "pop"},
    {"type": "label", "text": "Normal airway", "x": 500, "y": 300, "pill": True, "show": [0.02, 0.2], "enter": "pop"},
    {"type": "label", "text": "Muscles tighten", "x": 500, "y": 300, "pill": True, "color": "warning", "show": [0.22, 0.48], "enter": "pop"},
    {"type": "label", "text": "Mucus", "x": 760, "y": 360, "size": 36, "point_to": [560, 515], "show": [0.35, 0.7]},
    {"type": "label", "text": "Inhaler opens it", "x": 500, "y": 300, "pill": True, "color": "good", "show": [0.74, 1.0], "enter": "pop"},
    {"type": "particles", "style": "spark", "points": [[150, 820], [330, 700], [500, 640]], "count": 16, "size": 12, "color": "drug", "show": [0.7, 0.95], "speed": 1.5},
    {"type": "label", "text": "Medicine", "x": 200, "y": 880, "size": 34, "color": "drug", "show": [0.72, 1.0]},
    {"type": "counter", "x": 800, "y": 780, "size": 90, "suffix": "%", "text": "air", "anim": {"value": [[0.15, 100], [0.45, 38], [0.72, 38], [0.9, 95]]}},
]}


LAST_ERROR = {}  # project dir -> why the last automatic design failed (shown in the Check tab)


def _ask(prompt, max_tokens=6000, timeout=240):
    """Claude call with a long timeout (a full scene can take 1-2 minutes to write) and one retry."""
    import urllib.request, urllib.error
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key: return None
    body = json.dumps({"model": os.environ.get("MEDREEL_LLM", "claude-sonnet-5-5"), "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    err = None
    for attempt in range(2):
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", body,
                                     {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            if r.get("stop_reason") == "max_tokens": log.warning("animation design was cut off (max_tokens)")
            return "".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text")
        except urllib.error.HTTPError as e:
            detail = ""
            try: detail = e.read().decode()[:300]
            except Exception: pass
            err = RuntimeError(f"Claude HTTP {e.code}: {detail}")
            if e.code in (400, 401, 403, 404): break  # bad key / model / no credit: retrying won't help
        except Exception as e:
            err = e
    raise err


def _parse(txt):
    if not txt: return None
    try: return json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    except Exception: return None


def available(): return bool(os.environ.get("ANTHROPIC_API_KEY"))


def design(sentence, context="", lang="en", specialty="", idea=""):
    """One sentence (+ what is said around it) -> a validated scene, or None."""
    lname = {"ar": "Arabic", "fr": "French"}.get(lang, "English")
    prompt = (SPEC + "\n\nExample of a good scene (English):\n" + json.dumps(EXAMPLE, ensure_ascii=False) +
              f"\n\nThe doctor ({specialty or 'physician'}) says, in {lname}:\n\"{sentence}\"\n"
              + (f"Around it they also say: \"{context[:600]}\"\n" if context else "")
              + (f"Idea for the animation: {idea}\n" if idea else "")
              + f"Design the animation for that sentence. Labels and title in {lname}.")
    for attempt in range(2):
        raw = _ask(prompt)
        sc = validate(_parse(raw))
        if sc and len(sc["elements"]) >= 3: return sc
        log.warning("animation design unusable (attempt %d): %s", attempt + 1, (raw or "")[:300].replace("\n", " "))
    return None


def pick_moments(sentences, specialty="", want=2, lang="en"):
    """Claude picks the sentences that a medical animation explains best. sentences: [(i, text)] -> [{"i","idea"}]."""
    if not sentences or want <= 0: return []
    txt = _ask(f"A {specialty or 'physician'} says these numbered sentences in a short video:\n" + "\n".join(f"{i}: {t}" for i, t in sentences) +
               f"\n\nPick up to {want} sentences that would be explained best by a simple animated medical diagram (a body process, a "
               "disease mechanism, how a treatment works, a number or risk changing). Skip greetings, introductions and sign-offs, spread "
               'them through the video. Return ONLY JSON: [{"i": <number>, "idea": "<one sentence in English describing the animation>"}]', 800)
    try:
        if not txt: return []
        arr = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
        ok = {i for i, _ in sentences}
        out, seen = [], set()
        for x in arr:
            if isinstance(x, dict) and int(x.get("i", -1)) in ok and int(x["i"]) not in seen:
                seen.add(int(x["i"])); out.append({"i": int(x["i"]), "idea": _txt(x.get("idea"), 300)})
        return out[:want]
    except Exception:
        log.warning("animation moments: could not read Claude's answer: %s", (txt or "")[:300])
        return []


def save(scene, project_dir, accent=(46, 160, 255), lang="en"):
    d = os.path.join(project_dir, "anims"); os.makedirs(d, exist_ok=True)
    sid = uuid.uuid4().hex[:12]
    json.dump(scene, open(os.path.join(d, sid + ".json"), "w"), ensure_ascii=False)
    try:  # small preview frame for the storyboard
        fr = render_scene(scene, 540, 960, 3.6, 6.0, accent, lang)
        cv2.imwrite(os.path.join(d, sid + ".jpg"), cv2.cvtColor(cv2.resize(fr, (180, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    except Exception: log.exception("animation thumbnail")
    return sid


def _context(sents, i):
    return " ".join(sents[j]["text"] for j in (i - 1, i + 1) if 0 <= j < len(sents))


def create_for_sentence(plan, project_dir, i, idea=""):
    """Design + save one animation for sentence i. -> scene id (or None)."""
    sents = plan["analysis"]["sentences"]
    lang = plan.get("language", "en")
    sc = design(sents[i]["text"], _context(sents, i), lang, plan["branding"].get("specialty", ""), idea)
    if not sc: return None, None
    from .draw import hex2rgb
    try: acc = hex2rgb(plan["branding"].get("primary", "#2EA0FF"))
    except Exception: acc = (46, 160, 255)
    return save(sc, project_dir, acc, lang), sc["title"]


def auto(plan, project_dir, n=2, reserve=0):
    """Add up to n AI-designed animations to a new video, on sentences no other visual uses. -> number added."""
    if not available() or n <= 0: return 0
    import concurrent.futures as cf
    from . import commands as C
    presenter = plan.get("presenter", True)
    sents = plan["analysis"]["sentences"]
    busy = [(s["start"], s["end"]) for s in plan["scenes"] if s["graphic"]["asset"] not in ("icons",)]
    free = [(i, s["text"]) for i, s in enumerate(sents) if s["end"] - s["start"] >= 1.2 and s["start"] > (3.5 if presenter else 1.0)
            and not any(a < s["end"] - 0.3 and b > s["start"] + 0.3 for a, b in busy)]
    log.info("AI animations: %d free sentence(s) to choose from (keeping %d for AI images)", len(free), reserve)
    if reserve and len(free) > 1:  # leave moments for the AI illustrations so the video always gets a mix
        n = max(1, min(n, len(free) - reserve))
    if not free:
        _note(plan, "No free moment was left for an AI-designed animation — add one from the storyboard (click a sentence)."); return 0
    try:
        picks = pick_moments(free, plan["branding"].get("specialty", ""), n, plan.get("language", "en"))
    except Exception as e:
        _note(plan, f"AI-designed animations were skipped: {str(e)[:160]}"); log.warning("animation moments failed: %s", e); return 0
    if not picks:
        _note(plan, "Claude didn't pick moments for AI-designed animations — add one from the storyboard (click a sentence)."); return 0
    added = 0; errors = []
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(create_for_sentence, plan, project_dir, p["i"], p["idea"]): p["i"] for p in picks}
        for fu in cf.as_completed(futs):
            try:
                sid, title = fu.result()
                if not sid: errors.append("design unusable"); continue
                i = futs[fu]
                plan["scenes"] = [x for x in plan["scenes"] if not (x["graphic"]["asset"] == "icons" and x["start"] < sents[i]["end"] and x["end"] > sents[i]["start"])]
                msg = C.apply(plan, {"op": "add_gen", "scene": sid, "sentence": i, "title": title})
                if msg.startswith("Added"): added += 1
            except Exception as e:
                errors.append(str(e)[:160]); log.exception("AI animation failed")
    if errors and not added: _note(plan, f"AI-designed animations could not be made this time ({errors[0]}). Try “Create a medical animation here” in the storyboard.")
    return added


def _note(plan, msg):
    """A visible note in the Check tab (warn level) so failures are never silent."""
    plan.setdefault("analysis", {}).setdefault("qa", {}).setdefault("issues", []).append({"area": "animations", "level": "warn", "msg": msg})
    log.warning(msg)
